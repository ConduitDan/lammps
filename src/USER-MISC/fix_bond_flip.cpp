#include <mpi.h>
#include <cstdio>
#include <cmath>
#include "fix_bond_flip.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "comm.h"
#include "random_mars.h"
#include "error.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBondFlip::FixBondFlip(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg)
{
    // Check the number of arguments
    if (narg == 11) {
        consider_energy_change = false;
    }
    else if (narg == 12) {
        consider_energy_change = true;
    }
    else {
        error->all(FLERR, "Wrong number of arguments");
    }

    // Check the values of arguments
    nevery = utils::inumeric(FLERR, arg[3], false, lmp);
    r_every = utils::inumeric(FLERR, arg[4], false, lmp);
    num_attempts = utils::inumeric(FLERR, arg[5], false, lmp);
    blmin = utils::numeric(FLERR, arg[6], false, lmp);
    blmax = utils::numeric(FLERR, arg[7], false, lmp);
    btype = utils::inumeric(FLERR, arg[8], false, lmp);
    dtype = utils::inumeric(FLERR, arg[9], false, lmp);
    seed = utils::inumeric(FLERR, arg[10], false, lmp);
    if (consider_energy_change) {
        k = utils::numeric(FLERR, arg[11], false, lmp);
    }
    if (nevery <= 0 || r_every <= 0 || num_attempts <= 0 || blmin <= 0 || blmax <= blmin || btype <= 0 || dtype <= 0 || seed <= 0 || consider_energy_change && k <= 0) {
        error->all(FLERR, "Illegal arguments");
    }

    // Check conditions
    if (atom->molecular != 1) {
        error->all(FLERR, "Molecular system required");
    }
    if (force->newton != 1 || force->newton_pair != 1 || force->newton_bond != 1) {
        error->all(FLERR, "Newton's 3rd law required");
    }

    // Declare that neighbor list rebuilding may be needed
    force_reneighbor = 1;
    next_reneighbor = -1;

    // Initialize the Marsaglia RNG
    random_all = new RanMars(lmp, seed);
    random_one = new RanMars(lmp, seed+comm->me);

    // Initialize the counters
    r_flipped = 0;
    r_attempts = 0;
}

/* ---------------------------------------------------------------------- */

FixBondFlip::~FixBondFlip()
{
    delete random_all;
    delete random_one;
}

/* ---------------------------------------------------------------------- */

int FixBondFlip::setmask()
{
    int mask = 0;
    mask |= END_OF_STEP;
    return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondFlip::init()
{
    // Require special bonds = (1, 1, 1)
    if (force->special_lj[1] != 1.0 || force->special_lj[2] != 1.0 || force->special_lj[3] != 1.0)
        error->all(FLERR, "Special bonds lj (1, 1, 1) required");
}

/* ---------------------------------------------------------------------- */

void FixBondFlip::end_of_step()
{
    comm->forward_comm();

    r_attempts += num_attempts;
    for (int i = 0; i != num_attempts; ++i) {
        pick();
        if (flip()) {
            ++r_flipped;
        }
    }

    next_reneighbor = update->ntimestep + 1;

    if (update->ntimestep % r_every == 0) {
        if (comm->me == 0) {
            if (screen) {
                std::fprintf(screen, "Flipped %d/%d\n", r_flipped, r_attempts);
            }
            if (logfile) {
                std::fprintf(logfile, "Flipped %d/%d\n", r_flipped, r_attempts);
            }
        }
        r_flipped = 0;
        r_attempts = 0;
    }
}

/* ----------------------------------------------------------------------
    Broadcast the atom with the local index m from root, after which every
    process will know the local index of it
------------------------------------------------------------------------- */

void FixBondFlip::bcast_atom(int &m, const int root) const
{
    tagint tag;
    if (comm->me == root) {
        tag = atom->tag[m];
    }
    MPI_Bcast(&tag, 1, MPI_INT, root, world);
    if (comm->me != root) {
        m = atom->map(tag);
    }
}

/* ----------------------------------------------------------------------
    Compute the energy of a dihedral
------------------------------------------------------------------------- */

double FixBondFlip::compute_dihedral_energy(const int m1, const int m2, const int m3, const int m4) const
{
    // 1st bond

    const double vb1x = atom->x[m1][0] - atom->x[m2][0];
    const double vb1y = atom->x[m1][1] - atom->x[m2][1];
    const double vb1z = atom->x[m1][2] - atom->x[m2][2];

    // 2nd bond

    const double vb2x = atom->x[m3][0] - atom->x[m2][0];
    const double vb2y = atom->x[m3][1] - atom->x[m2][1];
    const double vb2z = atom->x[m3][2] - atom->x[m2][2];

    const double vb2xm = -vb2x;
    const double vb2ym = -vb2y;
    const double vb2zm = -vb2z;

    // 3rd bond

    const double vb3x = atom->x[m4][0] - atom->x[m3][0];
    const double vb3y = atom->x[m4][1] - atom->x[m3][1];
    const double vb3z = atom->x[m4][2] - atom->x[m3][2];

    // c,s calculation

    const double ax = vb1y*vb2zm - vb1z*vb2ym;
    const double ay = vb1z*vb2xm - vb1x*vb2zm;
    const double az = vb1x*vb2ym - vb1y*vb2xm;
    const double bx = vb3y*vb2zm - vb3z*vb2ym;
    const double by = vb3z*vb2xm - vb3x*vb2zm;
    const double bz = vb3x*vb2ym - vb3y*vb2xm;

    const double rasq = ax*ax + ay*ay + az*az;
    const double rbsq = bx*bx + by*by + bz*bz;
    if (rasq == 0 || rbsq == 0) {
        error->one(FLERR, "Cannot compute the dihedral energy");
    }

    const double ra2inv = 1.0/rasq;
    const double rb2inv = 1.0/rbsq;
    const double rabinv = std::sqrt(ra2inv*rb2inv);

    double c = (ax*bx + ay*by + az*bz)*rabinv;
    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;

    return k*(1+c);
}

/* ----------------------------------------------------------------------
    Find the dihedral of which the second and third atoms are m1 and m2
    , and put the atom local index to m, the dihedral index to n
------------------------------------------------------------------------- */

bool FixBondFlip::find_dihedral(const int m1, const int m2, int &m, int &n) const
{
    if (m1 >= 0 && m1 < atom->nlocal) {
        for (int i = 0; i != atom->num_dihedral[m1]; ++i) {
            if (atom->dihedral_atom3[m1][i] == atom->tag[m2]) {
                m = m1;
                n = i;
                return true;
            }
        }
    }
    return false;
}

/* ----------------------------------------------------------------------
    Find the dihedral of which the second and third atoms are m1 and m2
    or m2 and m1, and put the atom local index to m, the dihedral index
    to n
------------------------------------------------------------------------- */

bool FixBondFlip::find_dihedral_other(const int m1, const int m2, int &m, int &n) const
{
    if (find_dihedral(m1, m2, m, n)) {
        return true;
    } else {
        return find_dihedral(m2, m1, m, n);
    }
}

/* ----------------------------------------------------------------------
    Delete a bond
------------------------------------------------------------------------- */

void FixBondFlip::delete_bond(const int m, const int n) const
{
    for (int i = n; i != atom->num_bond[m]-1; ++i) {
        atom->bond_type[m][i] = atom->bond_type[m][i+1];
        atom->bond_atom[m][i] = atom->bond_atom[m][i+1];
    }
    --(atom->num_bond[m]);
}

/* ----------------------------------------------------------------------
    Create a bond
------------------------------------------------------------------------- */

void FixBondFlip::create_bond(const int m, const int m2) const
{
    if (atom->num_bond[m] == atom->bond_per_atom) {
        error->one(FLERR, "Not enough space for new bond");
    }
    atom->bond_type[m][atom->num_bond[m]] = btype;
    atom->bond_atom[m][atom->num_bond[m]] = atom->tag[m2];
    ++(atom->num_bond[m]);
}

/* ----------------------------------------------------------------------
    Delete a dihedral
------------------------------------------------------------------------- */

void FixBondFlip::delete_dihedral(const int m, const int n) const
{
    for (int i = n; i != atom->num_dihedral[m]-1; ++i) {
        atom->dihedral_type[m][i] = atom->dihedral_type[m][i+1];
        atom->dihedral_atom1[m][i] = atom->dihedral_atom1[m][i+1];
        atom->dihedral_atom2[m][i] = atom->dihedral_atom2[m][i+1];
        atom->dihedral_atom3[m][i] = atom->dihedral_atom3[m][i+1];
        atom->dihedral_atom4[m][i] = atom->dihedral_atom4[m][i+1];
    }
    --(atom->num_dihedral[m]);
}

/* ----------------------------------------------------------------------
    Create a dihedral
------------------------------------------------------------------------- */

void FixBondFlip::create_dihedral(const int m1, const int m2, const int m3, const int m4) const
{
    if (atom->num_dihedral[m2] == atom->dihedral_per_atom) {
        error->one(FLERR, "Not enough space for new dihedral");
    }
    atom->dihedral_type[m2][atom->num_dihedral[m2]] = dtype;
    atom->dihedral_atom1[m2][atom->num_dihedral[m2]] = atom->tag[m1];
    atom->dihedral_atom2[m2][atom->num_dihedral[m2]] = atom->tag[m2];
    atom->dihedral_atom3[m2][atom->num_dihedral[m2]] = atom->tag[m3];
    atom->dihedral_atom4[m2][atom->num_dihedral[m2]] = atom->tag[m4];
    ++(atom->num_dihedral[m2]);
}

/* ----------------------------------------------------------------------
    Pick a bond randomly
------------------------------------------------------------------------- */

void FixBondFlip::pick() const
{
    // Count bonds on each process
    int num_bonds = 0;
    for (int m = 0; m != atom->nlocal; ++m) {
        if (atom->mask[m]&groupbit) {
            num_bonds += atom->num_bond[m];
        }
    }
    int total_num_bonds;
    MPI_Allreduce(&num_bonds, &total_num_bonds, 1, MPI_INT, MPI_SUM, world);
    int *array_num_bonds = new int[comm->nprocs];
    MPI_Allgather(&num_bonds, 1, MPI_INT, array_num_bonds, 1, MPI_INT, world);

    // Pick a bond and find the process that owns it
    bool success = false;
    int target_bond_index = static_cast<int>(total_num_bonds*random_all->uniform());
    for (int i = 0; i != comm->nprocs; ++i) {
        if (target_bond_index < array_num_bonds[i]) {
            bproc = i;
            success = true;
            break;
        } else {
            target_bond_index -= array_num_bonds[i];
        }
    }
    if (!success) {
        error->all(FLERR, "Cannot find the process that owns the picked bond");
    }
    delete [] array_num_bonds;

    // Find the picked bond within the owner process
    if (comm->me == bproc) {
        bool success = false;
        for (int m = 0; m != atom->nlocal; ++m) {
            if (atom->mask[m]&groupbit) {
                if (target_bond_index < atom->num_bond[m]) {
                    bm = m;
                    bn = target_bond_index;
                    success = true;
                    break;
                } else {
                    target_bond_index -= atom->num_bond[m];
                }
            }
        }
        if (!success) {
            error->one(FLERR, "Cannot find the picked bond");
        }
    }
}

/* ----------------------------------------------------------------------
    Flip the picked bond
------------------------------------------------------------------------- */

bool FixBondFlip::flip() const
{
    int dm, dn;
    int dm1, dm2, dm3, dm4;

    if (comm->me == bproc) {

        // The other bond atom
        const int bm2 = atom->map(atom->bond_atom[bm][bn]);
        if (bm2 < 0) {
            error->one(FLERR, "Cannot find the other bond atom");
        }

        // Find the dihedral
        if (!find_dihedral(bm, bm2, dm, dn)) {
            error->one(FLERR, "Cannot find the dihedral");
        }

        // Retrieve the four atoms of the dihedral
        dm1 = atom->map(atom->dihedral_atom1[dm][dn]);
        dm2 = atom->map(atom->dihedral_atom2[dm][dn]);
        dm3 = atom->map(atom->dihedral_atom3[dm][dn]);
        dm4 = atom->map(atom->dihedral_atom4[dm][dn]);
        if (dm1 < 0 || dm2 < 0 || dm3 < 0 || dm4 < 0) {
            error->one(FLERR, "Cannot find dihedral atoms");
        }
        if (dm2 != dm) {
            error->one(FLERR, "Inconsistent dihedral");
        }
        if (dm2 != bm || dm3 != bm2) {
            error->one(FLERR, "Inconsistent bond and dihedral");
        }

    }

    // Check the distance
    {
        int status = 1;
        if (comm->me == bproc) {
            const double dx = atom->x[dm1][0] - atom->x[dm4][0];
            const double dy = atom->x[dm1][1] - atom->x[dm4][1];
            const double dz = atom->x[dm1][2] - atom->x[dm4][2];
            const double dsq = dx * dx + dy * dy + dz * dz;
            if (dsq <= blmin*blmin || dsq >= blmax*blmax) {
                status = 0;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, bproc, world);
        if (status == 0) {
            return false;
        }
    }

    bcast_atom(dm1, bproc);
    bcast_atom(dm2, bproc);
    bcast_atom(dm3, bproc);
    bcast_atom(dm4, bproc);

    // Check the opposing bond
    {
        int m_o, n_o;
        int status = 0;
        int count;
        if (find_dihedral_other(dm1, dm4, m_o, n_o)) {
            status = 1;
        }
        MPI_Allreduce(&status, &count, 1, MPI_INT, MPI_SUM, world);
        if (count != 0) {
            return false;
        }
    }

    const int in_mb1[] = {dm1, dm1, dm4, dm4};
    const int in_mb2[] = {dm2, dm3, dm2, dm3};
    const int in_mo[] = {dm3, dm2, dm3, dm2};
    const int in_mn[] = {dm4, dm4, dm1, dm1};
    bool own_in[4];
    int in_m[4], in_n[4];

    // Find involved dihedrals
    for (int i = 0; i != 4; ++i) {
        own_in[i] = find_dihedral_other(in_mb1[i], in_mb2[i], in_m[i], in_n[i]);
    }

    // Check the energy change
    if (consider_energy_change) {
        double in_delta = 0;
        for (int i = 0; i != 4; ++i) {
            if (own_in[i]) {
                const int in_dm1 = atom->map(atom->dihedral_atom1[in_m[i]][in_n[i]]);
                const int in_dm2 = atom->map(atom->dihedral_atom2[in_m[i]][in_n[i]]);
                const int in_dm3 = atom->map(atom->dihedral_atom3[in_m[i]][in_n[i]]);
                const int in_dm4 = atom->map(atom->dihedral_atom4[in_m[i]][in_n[i]]);
                if (in_dm1 == in_mo[i]) {
                    in_delta += compute_dihedral_energy(in_mn[i], in_dm2, in_dm3, in_dm4) - compute_dihedral_energy(in_dm1, in_dm2, in_dm3, in_dm4);
                    continue;
                }
                if (in_dm4 == in_mo[i]) {
                    in_delta += compute_dihedral_energy(in_dm1, in_dm2, in_dm3, in_mn[i]) - compute_dihedral_energy(in_dm1, in_dm2, in_dm3, in_dm4);
                    continue;
                }
                error->one(FLERR, "Inconsistent involved dihedral");
            }
        }
        double in_delta_total;
        MPI_Reduce(&in_delta, &in_delta_total, 1, MPI_DOUBLE, MPI_SUM, bproc, world);
        int status = 1;
        if (comm->me == bproc) {
            const double delta = compute_dihedral_energy(dm2, dm1, dm4, dm3) - compute_dihedral_energy(dm1, dm2, dm3, dm4) + in_delta_total;
            if (delta > 0 && random_one->uniform() > std::exp(-delta)) {
                status = 0;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, bproc, world);
        if (status == 0) {
            return false;
        }
    }

    // Fix involved dihedrals
    for (int i = 0; i != 4; ++i) {
        if (own_in[i]) {
            if (atom->dihedral_atom1[in_m[i]][in_n[i]] == atom->tag[in_mo[i]]) {
                atom->dihedral_atom1[in_m[i]][in_n[i]] = atom->tag[in_mn[i]];
                continue;
            }
            if (atom->dihedral_atom4[in_m[i]][in_n[i]] == atom->tag[in_mo[i]]) {
                atom->dihedral_atom4[in_m[i]][in_n[i]] = atom->tag[in_mn[i]];
                continue;
            }
            error->one(FLERR, "Inconsistent involved dihedral");
        }
    }

    if (comm->me == bproc) {
        delete_bond(bm, bn);
        delete_dihedral(dm, dn);
    }

    if (dm1 >= 0 && dm1 < atom->nlocal) {
        create_bond(dm1, dm4);
        create_dihedral(dm2, dm1, dm4, dm3);
    }

    return true;
}
