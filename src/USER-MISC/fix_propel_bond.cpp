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

/* ----------------------------------------------------------------------
   Contributing author: Matthew S. E. Peterson (Brandeis University)

   Thanks to Stefan Paquay (Brandeis) and Abhijeet Joshi (Brandeis) for
   implementation help and useful advice!
------------------------------------------------------------------------- */

#include "fix_propel_bond.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "random_mars.h"
#include "update.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* -------------------------------------------------------------------------- */

FixPropelBond::FixPropelBond(LAMMPS * lmp, int narg, char **argv)
  : Fix(lmp, narg, argv)
  , magnitude{0.0}
  , reversal_time{0.0}
  , reversal_prob{0.0}
  , nmolecules{0}
  , reversal_mode{OFF}
  , apply_to_type{nullptr}
  , reverse{nullptr}
  , random{nullptr}
{
  // skip 'fix', id, and group-id in argv
  argv += 3;
  narg -= 3;
  if (narg < 1) {
    error->all(FLERR, "Illegal fix propel/bond command");
  }

  magnitude = utils::numeric(FLERR, argv[0], false, lmp);

  // handle optional arguments
  int iarg = 1;
  std::string kw, val;
  while (iarg < narg) {
    kw = argv[iarg++];
    if (kw == "reverse") {
      reversal_mode = ON;
      
      if (narg - iarg < 2) {
        error->all(
            FLERR,
            "Error in fix propel/bond - too few values for 'reverse' keyword"
        );
      }
      
      reversal_time = utils::numeric(FLERR, argv[iarg++], false, lmp);
      if (reversal_time <= 0.0) {
        error->all(
            FLERR,
            "Error in fix propel/bond - reversal time must be positive"
        );
      }
      
      int seed = utils::inumeric(FLERR, argv[iarg++], false, lmp);
      random = new RanMars(lmp, seed);
    
    } else if (kw == "btypes") {
      apply_to_type = memory->create(
          apply_to_type, atom->nbondtypes + 1, "propel/bond:apply_to_type"
      );
      std::memset(apply_to_type, 0, (atom->nbondtypes + 1)*sizeof(int));
      
      int ok = 0;
      int ilo, ihi;
      while (iarg < narg) {
        val = argv[iarg++];
        if (!(std::isdigit(val[0]) || val[0] == '*')) {
          ok = 0;
          break;
        }
        
        utils::bounds(FLERR, val, 1, atom->nbondtypes, ilo, ihi, error);
        for (int type = ilo; type <= ihi; ++type) {
          apply_to_type[type] = 1;
        }
        
        ok = 1;
        iarg++;
      }

      if (!ok) {
        error->all(
            FLERR,
            "Error in fix propel/bond - Invalid bond type in 'btypes' keyword"
        );
      }

    } else {
      error->all(
          FLERR,
          fmt::format("Error in fix propel/bond - Unknown keyword '{}'", kw)
      );
    }
  }
}

/* -------------------------------------------------------------------------- */

FixPropelBond::~FixPropelBond()
{
  if (random) delete random;
  memory->destroy(reverse);
  memory->destroy(apply_to_type);
}

/* -------------------------------------------------------------------------- */

void FixPropelBond::init()
{
  if (reversal_mode == ON) {
    grow_reversal_list();
    reversal_prob = update->dt / reversal_time;
  }
}

/* -------------------------------------------------------------------------- */

int FixPropelBond::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* -------------------------------------------------------------------------- */

double FixPropelBond::memory_usage()
{
  double bytes = sizeof(FixPropelBond);
  if (apply_to_type) bytes += (atom->nbondtypes + 1.0) * sizeof(int);
  if (reverse) bytes += (nmolecules + 1.0) * sizeof(int);
  if (random) bytes += sizeof(RanMars);
  return bytes;
}

/* -------------------------------------------------------------------------- */

void FixPropelBond::pre_force(int /* vlag */)
{
  int i, j, type, sign;
  double dx, dy, dz, r, scale;
  tagint mol;

  int newton_bond = force->newton_bond;
  
  int nbonds = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;

  for (int n = 0; n < nbonds; ++n) {
    i = bondlist[n][0];
    j = bondlist[n][1];
    type = bondlist[n][2];

    if (mask[i] & mask[j] & groupbit) {
      if (apply_to_type && !apply_to_type[type]) continue;

      if (reversal_mode == OFF) {
        // forces always point from atom with small to atom with large tag
        sign = (atom->tag[i] > atom->tag[j]) ? 1 : -1;
      } else {
        mol = std::max(atom->molecule[i], atom->molecule[j]);
        if (mol > nmolecules) grow_reversal_list();
        sign = reverse[mol];
      }

      dx = x[i][0] - x[j][0];
      dy = x[i][1] - x[j][1];
      dz = x[i][2] - x[j][2];
      r = std::sqrt(dx*dx + dy*dy + dz*dz);

      if (r > 0.0) scale = 0.5 * sign * magnitude / r;
      else scale = 0.0;

      if (newton_bond || i < nlocal) {
        f[i][0] += scale * dx;
        f[i][1] += scale * dy;
        f[i][2] += scale * dz;
      }

      if (newton_bond || j < nlocal) {
        f[j][0] += scale * dx;
        f[j][1] += scale * dy;
        f[j][2] += scale * dz;
      }
    }
  }

  if (reversal_mode == ON) update_reversal_time();
}

/* -------------------------------------------------------------------------- */

void FixPropelBond::grow_reversal_list()
{
  tagint max_molecule = 0;
  for (int i = 0; i < atom->nlocal; ++i) {
    max_molecule = std::max(max_molecule, atom->molecule[i]);
  }

  {
    tagint max;
    MPI_Allreduce(&max_molecule, &max, 1, MPI_LMP_TAGINT, MPI_MAX, MPI_COMM_WORLD);
    max_molecule = max;
  }

  if (max_molecule > nmolecules) {
    reverse = memory->grow(reverse, max_molecule + 1, "propel/bond:reverse");
    for (int mol = nmolecules + 1; mol <= max_molecule; ++mol) {
      reverse[mol] = 1;
    }
    
    if (nmolecules == 0) reverse[0] = 1;
    nmolecules = max_molecule;
  }
}

/* -------------------------------------------------------------------------- */

void FixPropelBond::update_reversal_time()
{
  // reversal times are implicitly sampled from a geometric distribution, the
  // discrete analog of the exponential distribution, since it is memoryless
  for (int i = 1; i <= nmolecules; ++i) {
    if (random->uniform() <= reversal_prob) reverse[i] = -reverse[i];
  }
}
