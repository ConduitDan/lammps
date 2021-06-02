/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
   Contributed by Matthew S.E. Peterson @ Brandeis University

   Based on an initial implementation by Cong Qiao @ Brandeis.
   Thanks to Stefan Paquay @ Brandeis for help!
------------------------------------------------------------------------- */

#include "fix_fluidize_mesh.h"

#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "lammps.h"
#include "lmptype.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "random_mars.h"
#include "update.h"
#include "utils.h"

#define GROW_AMOUNT 4096

#define ENABLE_DEBUGGING 1

#define ERROR_ONE(...) error->one(FLERR, fmt::format(__VA_ARGS__))

#define ERROR_ALL(...) error->all(FLERR, fmt::format(__VA_ARGS__))

#define ILLEGAL(...)                                                           \
  error->all(FLERR,                                                            \
             fmt::format("Illegal fix fluidize/mesh command - " __VA_ARGS__))

#if defined(ENABLE_DEBUGGING) && ENABLE_DEBUGGING == 1
#define DEBUG(...) fmt::print(__VA_ARGS__)
#else
#define DEBUG(...)
#endif

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

struct FixFluidizeMesh::bond_type {
  tagint atoms[2];
  tagint type;
};

/* ---------------------------------------------------------------------- */

struct FixFluidizeMesh::dihedral_type {
  tagint atoms[4];
  tagint type;
};

/* ---------------------------------------------------------------------- */

FixFluidizeMesh::FixFluidizeMesh(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg), random(nullptr),
      temperature(nullptr), staged_swaps(nullptr), staged_swaps_all(nullptr),
      num_staged_swaps(0), max_staged_swaps(0), num_staged_swaps_all(0),
      max_staged_swaps_all(0), swap_probability(0.0), rmax2(0), rmin2(0),
      kbt(0) {
  if (narg < 6 || narg > 10) {
    ILLEGAL("incorrect number of arguments");
  }
  arg += 3;
  narg -= 3;

  nevery = utils::inumeric(FLERR, arg[0], false, lmp);
  if (nevery <= 0) {
    ILLEGAL("nevery must be positive");
  }

  swap_probability = utils::numeric(FLERR, arg[1], false, lmp);
  if (swap_probability < 0 || swap_probability > 1) {
    ILLEGAL("swap probability must be in the range [0, 1]");
  }

  int seed = utils::inumeric(FLERR, arg[2], false, lmp);
  random = new RanMars(lmp, seed + comm->me);

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "rmax") == 0) {
      if (iarg + 1 >= narg) {
        ILLEGAL("no value given to keyword 'rmax'");
      }
      double rmax = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (rmax <= 0) {
        ILLEGAL("value of 'rmax' must be positive");
      }
      rmax2 = rmax * rmax;
      iarg += 2;
    } else if (strcmp(arg[iarg], "rmin") == 0) {
      if (iarg + 1 >= narg) {
        ILLEGAL("no value given to keyword 'rmin'");
      }
      double rmin = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (rmin <= 0) {
        ILLEGAL("value of 'rmin' must be positive");
      }
      rmin2 = rmin * rmin;
      iarg += 2;
    } else {
      ILLEGAL("unknown keyword '{}'", arg[3]);
    }
  }

  if ((rmin2 > 0 || rmax2 > 0) && (rmin2 >= rmax2)) {
    ILLEGAL("Value of 'rmin' must be less than that of 'rmax'");
  }

  auto id_temp = id + std::string("_temp");
  modify->add_compute(id_temp + " all temp");

  int i = modify->find_compute(id_temp);
  if (i < 0) {
    ERROR_ALL("unable to compute system temperature");
  } else {
    temperature = modify->compute[i];
  }

  memory->create(staged_swaps, GROW_AMOUNT, "fix/fluidize/mesh:staged_swaps");
  max_staged_swaps = GROW_AMOUNT;
}

/* ---------------------------------------------------------------------- */

FixFluidizeMesh::~FixFluidizeMesh() {
  if (random) delete random;
  if (temperature) modify->delete_compute(temperature->id);
  memory->destroy(staged_swaps);
  memory->destroy(staged_swaps_all);
}

/* ---------------------------------------------------------------------- */

int FixFluidizeMesh::setmask() {
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::init() {
  if (atom->molecular != Atom::MOLECULAR) {
    ERROR_ALL("fix fluidize/mesh requires a molecular system");
  }

  if (!force->newton_bond) {
    ERROR_ALL("fix fluidize/mesh requires 'newton on'");
  }
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::post_integrate() {
  if (update->ntimestep % nevery != 0) return;

  int ndihedrals = neighbor->ndihedrallist;
  int **dihedrallist = neighbor->dihedrallist;

  comm->forward_comm();
  kbt = force->boltz * temperature->compute_scalar();

  bool skip;
  int a, b, c, d;
  bond_type old_bond, new_bond;
  dihedral_type dihedral;
  for (int i = 0; i < ndihedrals; ++i) {
    a = dihedrallist[i][0];
    b = dihedrallist[i][1];
    c = dihedrallist[i][2];
    d = dihedrallist[i][3];

    old_bond = {b, c};
    new_bond = {a, d};

    if (b < 0 || b >= atom->nlocal) continue;
    if (!(atom->mask[a] & atom->mask[b] & atom->mask[c] & atom->mask[d] & groupbit)) continue;
    if (random->uniform() >= swap_probability) continue;
    if (!find_bond(old_bond) || find_bond(new_bond)) continue;

    new_bond.type = old_bond.type;
    if (!accept_change(old_bond, new_bond)) continue;

    // convert to global atom tag before sending to other procs
    dihedral.atoms[0] = atom->tag[a];
    dihedral.atoms[1] = atom->tag[b];
    dihedral.atoms[2] = atom->tag[c];
    dihedral.atoms[3] = atom->tag[d];
    dihedral.type = dihedrallist[i][4];

    stage_swap(dihedral);
  }

  gather_swaps();
  commit();
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::stage_swap(dihedral_type dihedral) {
  if (num_staged_swaps == max_staged_swaps) {
    max_staged_swaps += GROW_AMOUNT;
    memory->grow(staged_swaps, max_staged_swaps,
                 "fix/fluidize/mesh:staged_swaps");
  }

  staged_swaps[num_staged_swaps++] = dihedral;
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::gather_swaps() {
  // this is a scale factor to correct the fact that we are sending/receiving
  // arrays of `dihedral_type` as if they were arrays of `tagint`s.
  constexpr int dihedral_size = sizeof(dihedral_type) / sizeof(tagint);

  int *layout = new int[2 * comm->nprocs];
  int *counts = layout;
  int *displs = layout + comm->nprocs;

  MPI_Allgather(&num_staged_swaps, 1, MPI_INT, counts, 1, MPI_INT,
                MPI_COMM_WORLD);

  displs[0] = 0;
  counts[0] *= dihedral_size;
  for (int i = 1; i < comm->nprocs; ++i) {
    counts[i] *= dihedral_size;
    displs[i] = (displs[i - 1] + counts[i - 1]);
  }

  num_staged_swaps_all = displs[comm->nprocs - 1] + counts[comm->nprocs - 1];
  if (num_staged_swaps_all > max_staged_swaps_all) {
    max_staged_swaps_all =
        GROW_AMOUNT * (1 + num_staged_swaps_all / GROW_AMOUNT);
    memory->grow(staged_swaps_all, max_staged_swaps_all,
                 "fix/fluidize/mesh:staged_swaps_all");
  }

  int send_size = num_staged_swaps * dihedral_size;
  MPI_Allgatherv(staged_swaps, send_size, MPI_LMP_TAGINT, staged_swaps_all,
                 counts, displs, MPI_LMP_TAGINT, MPI_COMM_WORLD);

  delete[] layout;
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::commit() {
  /*
  For each staged dihedral {a, b, c, d}, performs the following transformation

    e---a---f         e---a---f
     \ / \ /           \ /|\ / 
      b---c     --->    b | c    
     / \ / \           / \|/ \ 
    g---d---h         g---d---h

  This requires changing the bond {b, c} to the bond {a, d}, as well as the
  following 5 dihedral swaps:

    1. {a, b, c, d} -> {b, a, d, c}
    2. {e, a, b, c} -> {e, a, b, d}
    3. {f, a, c, b} -> {f, a, c, d}
    4. {g, b, d, c} -> {g, b, d, a}
    5. {h, c, d, b} -> {h, c, d, a}

  Note that, with exception to the first "main" swap, no other swap requires a
  change to the owning atom of the dihedral.
  */

  int indices[5];
  bond_type old_bond;
  bond_type new_bond;
  dihedral_type old_dihedrals[5];
  dihedral_type new_dihedrals[5];
  tagint a, b, c, d, e, f, g, h;
  for (int i = 0; i < num_staged_swaps_all; ++i) {
    a = staged_swaps_all[i].atoms[0];
    b = staged_swaps_all[i].atoms[1];
    c = staged_swaps_all[i].atoms[2];
    d = staged_swaps_all[i].atoms[3];    

    old_bond = {b, c};
    new_bond = {a, d};

    old_dihedrals[0] = {a, b, c, d};
    old_dihedrals[1] = {0, a, b, c};
    old_dihedrals[2] = {0, a, c, b};
    old_dihedrals[3] = {0, b, d, c};
    old_dihedrals[4] = {0, c, d, b};

    bool all_found = true;
    for (int i = 0; i < 5; ++i) {
      indices[i] = find_dihedral(old_dihedrals[i]);
      if (indices[i] < 0) all_found = false;
    }
    if (!all_found) continue;


    tagint e = find_exterior_atom(old_dihedrals[1], old_dihedrals[0]);
    tagint f = find_exterior_atom(old_dihedrals[2], old_dihedrals[0]);
    tagint g = find_exterior_atom(old_dihedrals[3], old_dihedrals[0]);
    tagint h = find_exterior_atom(old_dihedrals[4], old_dihedrals[0]);

    dihedral_type new0 = {b, a, d, c};
    dihedral_type new1 = {e, a, b, d};
    dihedral_type new2 = {f, a, c, d};
    dihedral_type new3 = {g, b, d, a};
    dihedral_type new4 = {h, c, d, a};

    remove_bond_at(old_bond, find_bond(old_bond));
    insert_bond(new_bond);

    for (int i = 0; i < 5; ++i) {
      remove_dihedral_at(old_dihedrals[i], indices[i]);
      insert_dihedral(new_dihedrals[i]);
    }
  }
}

/* ---------------------------------------------------------------------- */

// Attempts the following transformation of a dihedral:
//
//    a               a
//   /               /|
//  b---c    -->    b | c
//     /              |/
//    d               d
//
// That is, given an initially ordered dihedral {a, b, c, d}, it transforms
// into {b, a, d, c} by swapping the central bond ({b, c} -> {a, d}). This also
// requires updating the dihedrals that contain atoms within this dihedral.
void FixFluidizeMesh::try_swap(int a, int b, int c, int d) {
  bond_type old_bond = {b, c};
  bond_type new_bond = {a, d};
  if (!find_bond(old_bond) || find_bond(new_bond))
    return;
  else
    new_bond.type = old_bond.type;

  dihedral_type old_dihedral = {a, b, c, d};
  dihedral_type new_dihedral = {b, a, d, c};
  if (!find_dihedral(old_dihedral) || find_dihedral(new_dihedral))
    return;
  else
    new_dihedral.type = old_dihedral.type;

  if (!accept_change(old_bond, new_bond))
    return;

  dihedral_type to_insert[4];
  dihedral_type to_remove[] = {
      {-1, a, b, -1},
      {-1, c, d, -1},
      {-1, a, c, -1},
      {-1, b, d, -1},
  };
  int interior_atoms[] = {d, a, d, a};
  int exterior_atom;
  for (int i = 0; i < 4; ++i) {
    if (find_dihedral(to_remove[i]) < 0) {
      return;
    }

    exterior_atom = find_exterior_atom(to_remove[i], old_dihedral);
    if (exterior_atom < 0) {
      return;
    }

    a = exterior_atom;
    b = to_remove[i].atoms[1];
    c = to_remove[i].atoms[2];
    d = interior_atoms[i];
    to_insert[i] = {a, b, c, d};
  }

  for (int i = 0; i < 4; ++i) {
    swap_dihedrals(to_remove[i], to_insert[i]);
  }
  swap_dihedrals(old_dihedral, new_dihedral);

  // remove_bond(old_bond);
  // insert_bond(new_bond);

  next_reneighbor = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

bool FixFluidizeMesh::accept_change(bond_type old_bond, bond_type new_bond) {
  auto compute_energy = [&](bond_type bond) {
    int a = bond.atoms[0];
    int b = bond.atoms[1];

    double dx = atom->x[b][0] - atom->x[a][0];
    double dy = atom->x[b][1] - atom->x[a][1];
    double dz = atom->x[b][2] - atom->x[a][2];
    domain->minimum_image(dx, dy, dz);

    double f; // unused
    double r2 = dx * dx + dy * dy + dz * dz;

    double energy = 0.0;
    if (force->bond) {
      if ((rmax2 > 0 && r2 > rmax2) || (rmin2 > 0 && r2 < rmin2)) {
        energy += std::numeric_limits<double>::infinity();
      } else {
        energy += force->bond->single(bond.type, r2, a, b, f);
      }
    }

    return energy;
  };

  double delta = compute_energy(new_bond) - compute_energy(old_bond);
  return (delta < 0) || (random->uniform() < std::exp(-delta / kbt));
}

/* ---------------------------------------------------------------------- */

int FixFluidizeMesh::find_bond(bond_type &bond) {
  auto find = [&](tagint atag, tagint btag) {
    int a = atom->map(atag);
    if (a < 0) return -1;

    bond.atoms[0] = atag;
    bond.atoms[1] = btag;
    
    for (int i = 0; i < atom->num_bond[a]; ++i) {
      if (atom->bond_atom[a][i] == btag) {
        bond.type = atom->bond_type[a][i];
        return i;
      }
    }

    return -1;
  };

  tagint atag = bond.atoms[0];
  tagint btag = bond.atoms[1];
  auto index = find(atag, btag);
  if (index < 0) index = find(btag, atag);
  return index;
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::remove_bond_at(bond_type bond, int index) {
  tagint atag = bond.atoms[0];
  
  int a = atom->map(atag);
  if (a < 0) return;
  
  int &num_bonds = atom->num_bond[a];
  int *bond_type = atom->bond_type[a];
  tagint *bond_atom = atom->bond_atom[a];

  if (index < 0 || index >= num_bonds) {
    ERROR_ONE("attempted to remove bond that does not exist");
  }

  num_bonds--;
  if (index != num_bonds) {
    bond_atom[index] = bond_atom[num_bonds];
    bond_type[index] = bond_type[num_bonds];
  }
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::insert_bond(bond_type bond) {
  tagint atag = bond.atoms[0];
  tagint btag = bond.atoms[1];
  if (atag > btag) std::swap(atag, btag);

  int a = atom->map(atag);
  if (a < 0) return;

  int &num_bonds = atom->num_bond[a];
  if (num_bonds < atom->bond_per_atom) {
    atom->bond_atom[a][num_bonds] = btag;
    atom->bond_type[a][num_bonds] = bond.type;
    num_bonds++;
  } else {
    ERROR_ONE("No space for addition bonds - consider increasing "
              "extra/bond/per/atom");
  }
}

/* ---------------------------------------------------------------------- */

int FixFluidizeMesh::find_dihedral(dihedral_type &dihedral) {
  auto find = [&](tagint atag, tagint btag) {
    int a = atom->map(atag);
    if (a < 0) return -1;

    int num_dihedral = atom->num_dihedral[a];
    int *dtype = atom->dihedral_type[a];
    for (int i = 0; i < num_dihedral; ++i) {
      if (atom->dihedral_atom2[a][i] == atag &&
          atom->dihedral_atom3[a][i] == btag) {
        dihedral.atoms[0] = atom->dihedral_atom1[a][i];
        dihedral.atoms[1] = atom->dihedral_atom2[a][i];
        dihedral.atoms[2] = atom->dihedral_atom3[a][i];
        dihedral.atoms[3] = atom->dihedral_atom4[a][i];
        dihedral.type = atom->dihedral_type[a][i];
        return i;
      }
    }

    return -1;
  };

  tagint a = dihedral.atoms[1];
  tagint b = dihedral.atoms[2];
  int index = find(a, b);
  if (index < 0) find(b, a);
  return index;
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::remove_dihedral_at(dihedral_type dihedral, int index) {
  int datom = atom->map(dihedral.atoms[1]);
  if (datom < 0) return;

  int &num_dihedral = atom->num_dihedral[datom];
  int *dtype = atom->dihedral_type[datom];
  tagint *atom1 = atom->dihedral_atom1[datom];
  tagint *atom2 = atom->dihedral_atom2[datom];
  tagint *atom3 = atom->dihedral_atom3[datom];
  tagint *atom4 = atom->dihedral_atom4[datom];

  if (index < 0 || index >= num_dihedral) {
    ERROR_ONE("attempted to remove dihedral that does not exist");
  }

  num_dihedral--;
  if (index != num_dihedral) {
    dtype[index] = dtype[num_dihedral];
    atom1[index] = atom1[num_dihedral];
    atom2[index] = atom2[num_dihedral];
    atom3[index] = atom3[num_dihedral];
    atom4[index] = atom4[num_dihedral];
  }
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::insert_dihedral(dihedral_type dihedral) {
  tagint atag = dihedral.atoms[0];
  tagint btag = dihedral.atoms[1];
  tagint ctag = dihedral.atoms[2];
  tagint dtag = dihedral.atoms[3];

  int b = atom->map(btag);
  if (b < 0) return;

  int &num_dihedral = atom->num_dihedral[b];
  if (num_dihedral < atom->dihedral_per_atom) {
    atom->dihedral_atom1[b][num_dihedral] = atag;
    atom->dihedral_atom2[b][num_dihedral] = btag;
    atom->dihedral_atom3[b][num_dihedral] = ctag;
    atom->dihedral_atom4[b][num_dihedral] = dtag;
    atom->dihedral_type[b][num_dihedral] = dihedral.type;
    num_dihedral++;
  } else {
    ERROR_ONE("No space for addition dihedrals - consider increasing "
              "extra/dihedral/per/atom");
  }
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::swap_dihedrals(dihedral_type old_dihedral,
                                     dihedral_type new_dihedral) {
  // if (old_dihedral.atoms[1] != new_dihedral.atoms[1]) {
  //   remove_dihedral(old_dihedral);
  //   insert_dihedral(new_dihedral);
  //   return;
  // }

  // if the new dihedral is owned by the same atom as the old one, we can
  // optimize the removal/insertion
  int type = old_dihedral.type;
  // int index = old_dihedral.index;
  int a = new_dihedral.atoms[0];
  int b = new_dihedral.atoms[1];
  int c = new_dihedral.atoms[2];
  int d = new_dihedral.atoms[3];

  // atom->dihedral_atom1[b][index] = atom->tag[a];
  // atom->dihedral_atom2[b][index] = atom->tag[b];
  // atom->dihedral_atom3[b][index] = atom->tag[c];
  // atom->dihedral_atom4[b][index] = atom->tag[d];
  // atom->dihedral_type[b][index] = type;
}

/* ---------------------------------------------------------------------- */

bool FixFluidizeMesh::is_owned(dihedral_type dihedral) {
  auto id = atom->map(dihedral.atoms[1]);
  return 0 <= id && id < atom->nlocal;
}

/* ---------------------------------------------------------------------- */

// this effectively does a set difference between the atoms in 'a' and those
// in 'b'. Assumes that 3 of the 4 atoms in 'a' are in 'b'.
int FixFluidizeMesh::find_exterior_atom(dihedral_type a, dihedral_type b) {
  for (int i : a.atoms) {
    bool found = false;
    for (int j : b.atoms) {
      if (i == j) {
        found = true;
        break;
      }
    }

    if (!found)
      return i;
  }

  ERROR_ONE("unable to find exterior atom");
  return -1;
}
