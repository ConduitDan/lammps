/**
 * compute_active_atom.cpp
 *
 * Created by: Abhijeet Joshi
 * Maintained by: Matthew Peterson <matthew.se.peterson@gmail.com>
 */

#include <cmath>
#include <string>
#include <cstdlib>
#include "compute_active_atom.h"
#include "atom.h"
#include "atom_vec.h"
#include "molecule.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "bond.h" 
#include "pair.h"
#include "comm.h"
#include "memory.h"
#include "math_const.h"
#include "error.h"
#include "random_mars.h"
#include "domain.h"
#include "utils.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeActiveAtom::ComputeActiveAtom(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg),
    randphase{nullptr},
    poisson_delt{nullptr},
    next_time{nullptr},
    randtgt{nullptr}
{
    // this compute command should be given in a LAMMPS script as
    // >> compute <id> <group> active/atom <style> <force> <args...>
    // thus, the "arg" parameter should be accessed as
    //      id = arg[0]
    //      group = arg[1]
    //      active/atom = arg[2]
    //      style = arg[3]
    //      force = arg[4]
    //      if (style == "apolar" || style == "nematic") {
    //          reversal_time = arg[5]
    //          seed = arg[6]
    //      } else if (style != "polar") {
    //          ERROR
    //      }
    //          

    if (atom->avec->bonds_allow == 0) {
        error->all(FLERR,"Compute active/atom used when bonds are not allowed");
    }

    if (narg < 5) {
        error->all(FLERR,"Illegal compute active/atom command");
    } else {
        auto style = std::string{arg[3]};
        fstr = utils::numeric(FLERR, arg[4], false, lmp);
        if (style == "apolar" || style == "nematic") {
            omega = 1.0;
            if (narg < 7) {
                error->all(FLERR, "Illegal compute active/atom command");
            } else {
                tau = utils::numeric(FLERR, arg[5], false, lmp);
                seed = utils::inumeric(FLERR, arg[6], false, lmp);
            }
        } else if (style == "polar") {
            omega = 0.0;
            tau = 1.0;
            seed = 1;
        } else {
            error->all(FLERR, "Illegal compute active/atom command");
        }
    }

    // seed must be between 1 and 900,000,000 (inclusive)
    if (seed < 0) {
        seed = -seed;
    }
    seed = 1 + ((seed + comm->me) % 900000000);

    randtgt = new RanMars(lmp, seed);
    local_flag = 1;
    size_peratom_cols = 3;
    create_attribute = 1; 
    nvalues = narg - 3;
    size_local_cols = (nvalues == 1) ? 0 : nvalues;
    nvalues = 0;
    singleflag = 0;
    peratom_flag = 1;
    nmax = 0;
    maxmol = 0;
    factive = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputeActiveAtom::~ComputeActiveAtom()
{
    delete randtgt;

    memory->destroy(factive);
    memory->destroy(randphase);
    memory->destroy(poisson_delt);
    memory->destroy(next_time);
}

/* ---------------------------------------------------------------------- */

void ComputeActiveAtom::init()
{
    if (force->bond == nullptr) {
        error->all(FLERR,"No bond style is defined for compute active/atom");
    }
}

/* ---------------------------------------------------------------------- */

void ComputeActiveAtom::setup() {
    if (force->newton_bond) {
        error->all(FLERR, 
                   "compute active/atom requires 'newton off' to function");
    }
}

/* ---------------------------------------------------------------------- */

void ComputeActiveAtom::compute_peratom()
{
    int i, delt_steps, next_steps; 
    double MY_2PI=LAMMPS_NS::MathConst::MY_2PI;
    double next_rand;

    // no need to do all of this stochastic time reversal for polar activity!
    if (omega != 0.0) {
        tagint new_maxmol = 0;
        if (atom->molecule_flag) {
            for (i = 0; i < atom->nlocal; i++) {
                new_maxmol = MAX(atom->molecule[i], new_maxmol);
            }

            tagint maxmol_all;
            MPI_Allreduce(&new_maxmol, &maxmol_all, 1, 
                        MPI_LMP_TAGINT, MPI_MAX, world);
            new_maxmol = maxmol_all;
        }

        if (new_maxmol > maxmol) {
            randphase = memory->grow(randphase, new_maxmol,
                                    "compute/active/atom:randphase");
            poisson_delt = memory->grow(poisson_delt, new_maxmol,
                                        "compute/active/atom:poisson_delt");
            next_time = memory->grow(next_time, new_maxmol,
                                    "compute/active/atom:next_time");

            for (i = maxmol; i < new_maxmol; i++) {
                randphase[i] = 0.0;
                poisson_delt[i] = 0.0;
                next_time[i] = 0.0;
            }
        }
        maxmol = new_maxmol;
        
        if (comm->me == 0) { 
            for(i = 0; i < maxmol; i++) {
                poisson_delt[i] += static_cast<double>(update->dt);
                delt_steps = static_cast<int>(poisson_delt[i]/(update->dt));
                next_steps = static_cast<int>(next_time[i]/(update->dt))+1;

                if (delt_steps >= next_steps) {
                    randphase[i] = MY_2PI * (randtgt->uniform());
                    poisson_delt[i] = 0.0;
                    next_rand = randtgt->uniform();

                    if (next_rand < 1e-10) {
                        next_time[i] = 0.0;
                    } else {
                        next_time[i] = -std::log(next_rand)*tau;
                    }
                }  
            }
        }	

        MPI_Bcast(randphase, maxmol, MPI_DOUBLE, 0, world);
    }

    if (atom->nmax > nmax) {
        nmax = atom->nmax;
        memory->destroy(factive);
        memory->create(factive, nmax, size_peratom_cols, "compute/active/atom:factive");
        array_atom = factive;
    }

    int imol = -1;
    int nb, atom1, atom2, iatom, btype;
    tagint tagprev;
    double delx, dely, delz, rsq, delx1, dely1, delz1;

    double **x = atom->x;
    tagint *tag = atom->tag;
    tagint *molecule = atom->molecule; 
    int *num_bond = atom->num_bond;
    tagint **bond_atom = atom->bond_atom;
    int **bond_type = atom->bond_type;
    int *mask = atom->mask;

    int *molindex = atom->molindex;
    int *molatom = atom->molatom;
    Molecule **onemols = atom->avec->onemols;

    int nlocal = atom->nlocal;
    int molecular = atom->molecular;

    double phase;

    for (atom1 = 0; atom1 < nlocal; atom1++) {
        if (!(mask[atom1] & groupbit)) {
            continue;
        }

        if (molecular == 1) {
            nb = num_bond[atom1];
        } else {
            if (molindex[atom1] < 0) continue;
            imol = molindex[atom1];
            iatom = molatom[atom1];
            nb = onemols[imol]->num_bond[iatom];
        }

        delx = dely = delz = 0.0;
        for (i = 0; i < nb; i++) {
            if (molecular == 1) {
                btype = bond_type[atom1][i];
                atom2 = atom->map(bond_atom[atom1][i]);
            } else {
                tagprev = tag[atom1] - iatom - 1;
                btype = onemols[imol]->bond_type[atom1][i];
                atom2 = atom->map(onemols[imol]->bond_atom[atom1][i]+tagprev);
            }

            if (atom2 < 0 || !(mask[atom2] & groupbit)) continue;
            if (btype == 0) continue;
            
            // we simply assume that the "canonical" direction of active force
            // points from atoms with a smaller ID to atoms with a larger ID
            auto sign = (tag[atom1] > tag[atom2]) - (tag[atom1] < tag[atom2]);
            delx1 = sign * (x[atom1][0] - x[atom2][0]);
            dely1 = sign * (x[atom1][1] - x[atom2][1]);
            delz1 = sign * (x[atom1][2] - x[atom2][2]);

            // correctly account for minimum image convention
            domain->minimum_image(delx1,dely1,delz1);
            delx+=delx1;
            dely+=dely1;
            delz+=delz1;
        }

        // normalization procedure
        rsq = sqrt(delx*delx + dely*dely + delz*delz);
        if (rsq > 0){
            delx /= rsq;
            dely /= rsq;
            delz /= rsq;
        }
        
        phase = (omega == 0.0) ? 0.0 : randphase[molecule[atom1]-1];
        double temp = fstr * (2.0 * (1.0 - std::signbit(std::sin(phase)) * omega) - 1.0);
        
        factive[atom1][0] = temp*delx;  
        factive[atom1][1] = temp*dely;  
        factive[atom1][2] = temp*delz;
    }

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
   ------------------------------------------------------------------------- */

double ComputeActiveAtom::memory_usage()
{
    double bytes = nmax * size_peratom_cols * sizeof(double);
    return bytes;
}
