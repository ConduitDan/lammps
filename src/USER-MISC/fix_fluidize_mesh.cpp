/* -------atom-------------------atom--------------------------------------------
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
#include "dihedral.h"
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
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#define GROW_AMOUNT 4096

#define ENABLE_DEBUGGING 1

#define ERROR_ONE(...) \
  error->one(FLERR, fmt::format(__VA_ARGS__))

#define ERROR_ALL(...) \
  error->all(FLERR, fmt::format(__VA_ARGS__))

#define ILLEGAL(...) \
  error->all(FLERR, fmt::format("Illegal fix fluidize/mesh command - " __VA_ARGS__))

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
  int type;
  int index;
};

/* ---------------------------------------------------------------------- */

struct FixFluidizeMesh::dihedral_type {
  tagint atoms[4];
  int type;
  int index;
};

/* ---------------------------------------------------------------------- */

FixFluidizeMesh::FixFluidizeMesh(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg),
    random(nullptr),
    temperature(nullptr),
    swap_probability{},
    rmax2{},
    rmin2{},
    kbt{}
{
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
}

/* ---------------------------------------------------------------------- */

FixFluidizeMesh::~FixFluidizeMesh()
{
  if (random) delete random;
  if (temperature) modify->delete_compute(temperature->id);
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

  if(!utils::strmatch(force->dihedral_style,"^harmonic")) {
    ERROR_ALL("fix fluidize/mesh requires harmonic dihedral style");
  }
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::post_integrate() {
  if (update->ntimestep % nevery != 0) return;
  if (update->ntimestep % 10000 == 0) print_p_acc();

  comm->forward_comm();
  kbt = force->boltz * temperature->compute_scalar();

  bool skip;
  int a, b, c, d;

  //one sweep = ndihedrals attempts to flip bonds
  
  //std::cout << "Num dihedrals: " << atom->ndihedrals << std::endl;

  int cnt = 0;
  while(cnt<atom->ndihedrals){
    //Choose a dihedral at random; i = [0, ndihedral - 1]
    //Map dihedrals to atom and dihedral index
    /*
  std::map<int, std::pair<int, int>> dihedral_map;
 
  int dihedral_cnt=0;
  for (int i_det = 0; i_det < atom->nlocal; ++i_det) {
    if (!(atom->mask[i_det] & groupbit)) continue;
    if(atom->num_dihedral[i_det]==0) continue;
    for (int j = 0; j < atom->num_dihedral[i_det]; ++j){
      if (atom->dihedral_atom2[i_det][j] != atom->tag[i_det]) continue;
      dihedral_map[dihedral_cnt] = {i_det, j};
      dihedral_cnt++;
    }
  }
  */
    //For now, do this in an inefficient way -- just loop through atoms until you find the atom + dihedral indices corresponding to the randomly selected index for the dihedral list
    int index = random->integer(atom->ndihedrals);
    
    int i=-1;
    int j=-1;
    int dihedral_cnt=0;
    for (int i_det = 0; i_det < atom->nlocal; ++i_det) {
      //std::cout << "i : " << i_det << std::endl;
      if (!(atom->mask[i_det] & groupbit)) continue;
      if(atom->num_dihedral[i_det]==0) continue;
      for (int j_det = 0; j_det < atom->num_dihedral[i_det]; ++j_det){
        //std::cout << "j: " << j_det << std::endl;
        if (atom->dihedral_atom2[i_det][j_det] != atom->tag[i_det]) continue;
        if(dihedral_cnt==index){
          i = i_det;
          j = j_det;
          //std::cout << "i: " << i << " j: " << std::endl;
        }
        dihedral_cnt++;
      }
    }
    //std::cout << "index " << index << " i: " << i << " j: " << j << std::endl;
    //int i = dihedral_map[index].first;
    //int j = dihedral_map[index].second;

    //This is like an attempt frequency -- sets overall rate
    if (random->uniform() > swap_probability){
      cnt ++;
      continue;
    }  

    a = atom->map(atom->dihedral_atom1[i][j]);
    b = atom->map(atom->dihedral_atom2[i][j]);
    c = atom->map(atom->dihedral_atom3[i][j]);
    d = atom->map(atom->dihedral_atom4[i][j]);

    skip = false;
    for (int id : {a, b, c, d}) {
      if (id < 0) ERROR_ONE("fix fluidize/mesh needs a larger communication cutoff!");
        
      if ((id >= atom->nlocal) || !(atom->mask[id] & groupbit)) {
        skip = true;
        break;
      }
    }

    //Make an attempt to flip the bond
    //std::cout<<a<<", "<<b<<", "<<c<<", "<<d<<std::endl;
    //if (!skip) try_swap(a, b, c, d);
    if (!skip) {
      //std::cout<<"1. trying swap "<<a<<" "<<b<<" "<<c<<" "<<d<<std::endl;
      try_swap(a, b, c, d);
    }
    else {
      //std::cout<<"1. skipped"<<std::endl;
    }
    
    cnt += 1;
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
  if (!find_bond(old_bond) || find_bond(new_bond)){
    std::cout << "Error: could not find bonds to swap! (in try_swap)" << std::endl;
    return;
  } 
  else new_bond.type = old_bond.type;
  //std::cout<<"2. Bond found "<<find_bond(old_bond)<<" "<<find_bond(new_bond)<<std::endl;

  dihedral_type old_dihedral = {a, b, c, d};
  dihedral_type new_dihedral = {b, a, d, c};
  if (!find_dihedral(old_dihedral) || find_dihedral(new_dihedral)){
    std::cout << "Error: could not find dihedrals to swap! (in try_swap)" << std::endl;
    return;
  }
  else new_dihedral.type = old_dihedral.type;
  //std::cout<<"3. Dihedral found "<<find_dihedral(old_dihedral)<<" "<<find_dihedral(new_dihedral)<<std::endl;

  dihedral_type to_insert[4];
  dihedral_type to_remove[] = {
    {-1, a, b, -1},
    {-1, c, d, -1},
    {-1, a, c, -1},
    {-1, b, d, -1},
  };
  //std::cout<<"5. Swapping: - "<<std::endl;
  int interior_atoms[] = {d, a, d, a};
  int exterior_atom;
  //this apparently does not work when there is only one dihedral
  for (int i = 0; i < 4; ++i) {
    if (!find_dihedral(to_remove[i])){
      std::cout << "Error: could not find dihedral to remove!" << std::endl;
      return;
    } 
    //std::cout<<"5. dihedral found "<<i<<std::endl;

    exterior_atom = find_exterior_atom(to_remove[i], old_dihedral);
    //std::cout<<"5. Exterior atom found: "<<exterior_atom<<std::endl;
    if (exterior_atom < 0){
      std::cout << "Error: could not find exterior atom!" << std::endl;
      return;
    } 
    a = exterior_atom;
    b = to_remove[i].atoms[1];
    c = to_remove[i].atoms[2];
    d = interior_atoms[i];
    //std::cout<<"5. New dihedral: "<<a<<" "<<b<<" "<<c<<" "<<d<<" "<<std::endl;
    to_insert[i] = {a, b, c, d};
    to_insert[i].type = to_remove[i].type;
  }

  if (!accept_change(old_bond, new_bond, old_dihedral, new_dihedral, to_insert, to_remove)) {
    //std::cout<<"4. Bond change rejected"<<std::endl;
    n_reject++;
    return;
  }
  n_accept++;
  //std::cout << "Swapping bond\n";
  
  for (int i = 0; i < 4; ++i) {
    //std::cout<<"5. Swapping dihedral: "<<i<<std::endl;
    swap_dihedrals(to_remove[i], to_insert[i]);    
  }
  swap_dihedrals(old_dihedral, new_dihedral);
  
  remove_bond(old_bond);
  insert_bond(new_bond);

  //Check that new bond has acceptable length
  int a1 = new_bond.atoms[0];
  int b1 = new_bond.atoms[1];
  double dx = atom->x[b1][0] - atom->x[a1][0];
  double dy = atom->x[b1][1] - atom->x[a1][1];
  double dz = atom->x[b1][2] - atom->x[a1][2];
  domain->minimum_image(dx, dy, dz);
  double r2 = dx*dx + dy*dy + dz*dz;
  if (r2 < rmin2) {
    std::cout << "Error: accepted bond too short!" << std::endl;
  }
  if (r2 > rmax2) {
    std::cout << "Error: accepted bond too long!" << std::endl;
  }

  next_reneighbor = update->ntimestep;


/* ---------------------------------------------------------------------- */}
//print acceptance probability

void FixFluidizeMesh::print_p_acc() {
  std::cout << "No. swaps accepted: " << n_accept << std::endl;
  std::cout << "No. swaps rejected: " << n_reject << std::endl;
  std::cout << "Acceptance ratio: " << (1.0*n_accept)/(n_accept + n_reject) << std::endl;
}

/* ---------------------------------------------------------------------- */

double FixFluidizeMesh::compute_bending_energy(dihedral_type dihedral) {
  //Get positions of atoms in dihedral. Note: b and c are bonded atoms.
  int a = dihedral.atoms[0];
  int b = dihedral.atoms[1];
  int c = dihedral.atoms[2];
  int d = dihedral.atoms[3];

  double xa = atom->x[a][0];
  double ya = atom->x[a][1];
  double za = atom->x[a][2];

  double xb = atom->x[b][0];
  double yb = atom->x[b][1];
  double zb = atom->x[b][2];

  double xc = atom->x[c][0];
  double yc = atom->x[c][1];
  double zc = atom->x[c][2];

  double xd = atom->x[d][0];
  double yd = atom->x[d][1];
  double zd = atom->x[d][2];

  //Calculate angle theta which is the angle bw the two planes made by {a,c,b} and {b,c,d}

  double dx_cb = xc - xb;
  double dy_cb = yc - yb;
  double dz_cb = zc - zb;

  double dx_ab = xa - xb;
  double dy_ab = ya - yb;
  double dz_ab = za - zb;

  double dx_dc = xd - xc;
  double dy_dc = yd - yc;
  double dz_dc = zd - zc;

  double N1_x = (dy_cb * dz_dc) - (dz_cb * dy_dc);
  double N1_y = (dz_cb * dx_dc) - (dx_cb * dz_dc);
  double N1_z = (dx_cb * dy_dc) - (dy_cb * dx_dc);

  double N2_x = (-dy_cb * dz_ab) - (-dz_cb * dy_ab);
  double N2_y = (-dz_cb * dx_ab) - (-dx_cb * dz_ab);
  double N2_z = (-dx_cb * dy_ab) - (-dy_cb * dx_ab);

  double norm_N1 = std::sqrt(N1_x*N1_x + N1_y*N1_y + N1_z*N1_z);
  double norm_N2 = std::sqrt(N2_x*N2_x + N2_y*N2_y + N2_z*N2_z);

  N1_x = N1_x/norm_N1;
  N1_y = N1_y/norm_N1;
  N1_z = N1_z/norm_N1;

  N2_x = N2_x/norm_N2;
  N2_y = N2_y/norm_N2;
  N2_z = N2_z/norm_N2;
  
  double costheta = N1_x*N2_x + N1_y*N2_y + N1_z*N2_z;

  //Returns the energy associated with the dihedral
  return force->dihedral->single(dihedral.type, acos(costheta));
  
}
bool FixFluidizeMesh::accept_change(bond_type old_bond, bond_type new_bond, dihedral_type old_dihedral, dihedral_type new_dihedral, dihedral_type (&old_nbs)[4], dihedral_type (&new_nbs)[4]) {
  auto compute_bond_energy = [&](bond_type bond) {
    int a = bond.atoms[0];
    int b = bond.atoms[1];

    double dx = atom->x[b][0] - atom->x[a][0];
    double dy = atom->x[b][1] - atom->x[a][1];
    double dz = atom->x[b][2] - atom->x[a][2];
    domain->minimum_image(dx, dy, dz);

    double f; // unused
    double r2 = dx*dx + dy*dy + dz*dz;
    //std::cout<<"4. bond length: "<<r2<<std::endl;

    double energy = 0.0;
    if (force->bond) {
      //std::cout<<"4. Condition1 Met :"<<force->bond<<std::endl;
      if ((rmax2 > 0 && r2 > rmax2) || (rmin2 > 0 && r2 < rmin2))
        {
          //std::cout<<"4. Condition2 not Met - Inf "<<std::endl;
          //std::cout << "Proposed dihedral swap bond energy too high: bond length = " << sqrt(r2) << std::endl;
          energy += std::numeric_limits<double>::infinity();
        }
            else
        {
          energy += force->bond->single(bond.type, r2, a, b, f);
          //std::cout<<"3. Condition2 Met: "<<energy<<std::endl;
        }
    }

    //std::cout<<"4. Energy: "<<energy<<std::endl;
    return energy;
  };

  //Note that right now, this only works if the dihedrals are harmonic
  double energy_oldDihedrals = compute_bending_energy(old_dihedral);
  double energy_newDihedrals = compute_bending_energy(new_dihedral);

  //Check whether this works if fewer than 4 neighboring triangles
  for (int i = 0; i < 4; ++i) {
    //std::cout<<"5. Swapping dihedral: "<<i<<std::endl;
    energy_oldDihedrals += compute_bending_energy(old_nbs[i]);
    energy_newDihedrals += compute_bending_energy(new_nbs[i]);
  }

  double delta = compute_bond_energy(new_bond) + energy_newDihedrals - compute_bond_energy(old_bond) - energy_oldDihedrals;
  //std::cout<<"4. Energy difference in bonds: "<<delta<<std::endl;

  //Check length of new bond
  int a = new_bond.atoms[0];
  int b = new_bond.atoms[1];
  double dx = atom->x[b][0] - atom->x[a][0];
  double dy = atom->x[b][1] - atom->x[a][1];
  double dz = atom->x[b][2] - atom->x[a][2];
  domain->minimum_image(dx, dy, dz);
  double r2 = dx*dx + dy*dy + dz*dz;

  if (random->uniform() < std::exp(-delta / kbt)) {
    if (r2 < rmin2) {
      std::cout << "Error: accepted bond too short!" << std::endl;
    }
    if (r2 > rmax2) {
      std::cout << "Error: accepted bond too long!" << std::endl;
    }
    return true;
  }
  else {
    return false;
  }

  //return (delta < 0) || (random->uniform() < std::exp(-delta / kbt));
}

/* ---------------------------------------------------------------------- */

bool FixFluidizeMesh::find_bond(bond_type &bond) {
  auto find = [&](int a, int b) {
    bond.atoms[0] = a;
    bond.atoms[1] = b;
    tagint target = atom->tag[b];
    for (int i = 0; i < atom->num_bond[a]; ++i) {
      if (atom->bond_atom[a][i] == target) {
        bond.type = atom->bond_type[a][i];
        bond.index = i;
        return true;
      }
    }

    bond.index = -1;
    return false;
  };

  int a = bond.atoms[0];
  int b = bond.atoms[1];
  return find(a, b) || find(b, a);
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::remove_bond(bond_type bond) {
  int a = bond.atoms[0];
  int index = bond.index;
  int &num_bonds = atom->num_bond[a];
  int *bond_type = atom->bond_type[a];
  tagint *bond_atom = atom->bond_atom[a];
  
  if (index < 0 || num_bonds == 0) {
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
  int a = bond.atoms[0];
  int b = bond.atoms[1];
  if (atom->tag[a] > atom->tag[b]) std::swap(a, b);

  int &num_bonds = atom->num_bond[a];
  if (num_bonds < atom->bond_per_atom) {
    atom->bond_atom[a][num_bonds] = atom->tag[b];
    atom->bond_type[a][num_bonds] = bond.type;
    num_bonds++;
  } else {
    ERROR_ONE("No space for addition bonds - consider increasing extra/bond/per/atom");
  }
}

/* ---------------------------------------------------------------------- */

bool FixFluidizeMesh::find_dihedral(dihedral_type &dihedral) {
  auto find = [&](int a, int b) {
    int num_dihedral = atom->num_dihedral[a];
    int *dtype = atom->dihedral_type[a];
    
    for (int i = 0; i < num_dihedral; ++i) {
      if (atom->dihedral_atom2[a][i] == atom->tag[a] && 
          atom->dihedral_atom3[a][i] == atom->tag[b])
	{
	  dihedral.atoms[0] = atom->map(atom->dihedral_atom1[a][i]);
	  dihedral.atoms[1] = atom->map(atom->dihedral_atom2[a][i]);
	  dihedral.atoms[2] = atom->map(atom->dihedral_atom3[a][i]);
	  dihedral.atoms[3] = atom->map(atom->dihedral_atom4[a][i]);
	  dihedral.type = atom->dihedral_type[a][i];
	  
	  for (int id : dihedral.atoms) {
	    if (id < 0) ERROR_ONE("fix fluidize/mesh needs a larger communication cutoff!");
	  }
	  
	  dihedral.index = i;
	  return true;
	}
    }

    dihedral.index = -1;
    return false;
  };

  int a = dihedral.atoms[1];
  int b = dihedral.atoms[2];
  return find(a, b) || find(b, a);
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::remove_dihedral(dihedral_type dihedral) {
  int datom = dihedral.atoms[1]; //owner atom of dihedral
  int index = dihedral.index;
  
  //aliases
  int &num_dihedral = atom->num_dihedral[datom];
  int *dtype = atom->dihedral_type[datom];
  tagint *atom1 = atom->dihedral_atom1[datom];
  tagint *atom2 = atom->dihedral_atom2[datom];
  tagint *atom3 = atom->dihedral_atom3[datom];
  tagint *atom4 = atom->dihedral_atom4[datom];
  
  if (index < 0 || num_dihedral == 0 || index >= num_dihedral) {
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
  int a = dihedral.atoms[0];
  int b = dihedral.atoms[1];
  int c = dihedral.atoms[2];
  int d = dihedral.atoms[3];

  int &num_dihedral = atom->num_dihedral[b];
  if (num_dihedral < atom->dihedral_per_atom) {
    atom->dihedral_atom1[b][num_dihedral] = atom->tag[a];
    atom->dihedral_atom2[b][num_dihedral] = atom->tag[b];
    atom->dihedral_atom3[b][num_dihedral] = atom->tag[c];
    atom->dihedral_atom4[b][num_dihedral] = atom->tag[d];
    atom->dihedral_type[b][num_dihedral] = dihedral.type;
    num_dihedral++;
  } else {
    ERROR_ONE("No space for addition dihedrals - consider increasing extra/dihedral/per/atom");
  }
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::swap_dihedrals(dihedral_type old_dihedral, dihedral_type new_dihedral) {
  if (old_dihedral.atoms[1] != new_dihedral.atoms[1]) {

    //std::cout<<"6. 1 != 1"<<std::endl;
    remove_dihedral(old_dihedral);
    insert_dihedral(new_dihedral);
    return;
  }

  // if the new dihedral is owned by the same atom as the old one, we can
  // optimize the removal/insertion
  int type = old_dihedral.type;
  int index = old_dihedral.index;
  int a = new_dihedral.atoms[0];
  int b = new_dihedral.atoms[1];
  int c = new_dihedral.atoms[2];
  int d = new_dihedral.atoms[3];

  atom->dihedral_atom1[b][index] = atom->tag[a];
  atom->dihedral_atom2[b][index] = atom->tag[b];
  atom->dihedral_atom3[b][index] = atom->tag[c];
  atom->dihedral_atom4[b][index] = atom->tag[d];
  atom->dihedral_type[b][index] = type;
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

    if (!found) return i;
  }

  ERROR_ONE("unable to find exterior atom");
  return -1;
}
