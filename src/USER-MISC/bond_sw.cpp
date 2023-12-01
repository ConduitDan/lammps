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
   Contributed by Matthew S.E. Peterson @ Brandeis University
------------------------------------------------------------------------- */

#include "bond_sw.h"

#include <cmath>
#include <cstring>

#include "atom.h"
#include "neighbor.h"
#include "comm.h"
#include "update.h"
#include "force.h"
#include "memory.h"
#include "error.h"

#define ERROR_ALL(...) error->all(FLERR, fmt::format(__VA_ARGS__))
#define ERROR_ONE(...) error->one(FLERR, fmt::format(__VA_ARGS__))
#define WARN(...) error->warning(FLERR, fmt::format(__VA_ARGS__), 0)
#define LOGWARN(...) error->warning(FLERR, fmt::format(__VA_ARGS__), 1)

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondStillingerWeber::BondStillingerWeber(LAMMPS *lmp)
  : Bond(lmp),
  k{nullptr},
  lmin{nullptr},
  lmax{nullptr},
  delta{nullptr},
  sigma{nullptr}
{}

/* ---------------------------------------------------------------------- */

BondStillingerWeber::~BondStillingerWeber()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(lmin);
    memory->destroy(lmax);
    memory->destroy(delta);
    memory->destroy(sigma);
  }
}

/* ---------------------------------------------------------------------- */

void BondStillingerWeber::compute(int eflag, int vflag)
{
  int i1,i2,n,type;
  double a, b, c, u, l0, l1, l2, l3;
  double dx, dy, dz, r2, r;
  double ebond, fbond;

  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nbondlist; n++) {
    fbond = 0.0;
    ebond = 0.0;
    
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    dx = x[i1][0] - x[i2][0];
    dy = x[i1][1] - x[i2][1];
    dz = x[i1][2] - x[i2][2];

    r2 = dx*dx + dy*dy + dz*dz;
    r = std::sqrt(r2);

    l0 = lmin[type];
    l3 = lmax[type];
    l1 = l0 + delta[type];
    l2 = l3 - delta[type];    

    if (r < l1) {
      if (r <= l0) {
        LOGWARN("SW bond too short at step {}: {} ({}, {})",
          update->ntimestep, r, atom->tag[i1], atom->tag[i2]
        );
        if (r <= 0.5 * l0) ERROR_ONE("Bad SW bond");
        else r = 0.95 * l0 + 0.05 * l1;
      }
      
      a = sigma[type] / (r - l0);
      b = sigma[type] / (r - l1);
      c = std::exp(b);
      u = k[type] * a * c;

      fbond += u * (b * b + a) / (sigma[type] * r);
      if (eflag) {
        ebond += u;
      }
    }

    if (r > l2) {
      if (r >= l3) {
        LOGWARN("SW bond too long at step {}: {} ({}, {})",
          update->ntimestep, r, atom->tag[i1], atom->tag[i2]
        );
        if (r >= 2.0 * l3) ERROR_ONE("Bad SW bond");
        else r = 0.95 * l3 + 0.05 * l2;
      }
      
      a = sigma[type] / (l3 - r);
      b = sigma[type] / (l2 - r);
      c = std::exp(b);
      u = k[type] * a * c;

      fbond -= u * (b * b + a) / (sigma[type] * r);
      if (eflag) {
        ebond += u;
      }
    }

    // apply force to each of 2 atoms
    if (newton_bond || i1 < nlocal) {
      f[i1][0] += fbond * dx;
      f[i1][1] += fbond * dy;
      f[i1][2] += fbond * dz;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= fbond * dx;
      f[i2][1] -= fbond * dy;
      f[i2][2] -= fbond * dz;
    }

    if (evflag) ev_tally(i1, i2, nlocal, newton_bond, ebond, fbond, dx, dy, dz);
  }
}

/* ---------------------------------------------------------------------- */

void BondStillingerWeber::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(k,n+1,"bond:k");
  memory->create(lmin,n+1,"bond:lmin");
  memory->create(lmax,n+1,"bond:lmax");
  memory->create(delta,n+1,"bond:delta");
  memory->create(sigma,n+1,"bond:sigma");
  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

void BondStillingerWeber::coeff(int narg, char **arg)
{
  if (narg != 5 && narg != 6) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nbondtypes,ilo,ihi,error);

  double k_one = utils::numeric(FLERR,arg[1],false,lmp);
  double lmin_one = utils::numeric(FLERR,arg[2],false,lmp);
  double lmax_one = utils::numeric(FLERR,arg[3],false,lmp);
  double delta_one = utils::numeric(FLERR,arg[4],false,lmp);
  double sigma_one = 1.0;
  if (narg == 6) sigma_one = utils::numeric(FLERR,arg[5],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    lmin[i] = lmin_one;
    lmax[i] = lmax_one;
    delta[i] = delta_one;
    sigma[i] = sigma_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   check if special_bond settings are valid
------------------------------------------------------------------------- */

void BondStillingerWeber::init_style()
{
  // special bonds should be 0 1 1
  //(exclude pair interactions between directly bonded atoms,
  //but not between next-nearest and next-next-nearest neighbors.)

  if (force->special_lj[1] != 0.0 || force->special_lj[2] != 1.0 ||
      force->special_lj[3] != 1.0) {
    if (comm->me == 0)
      error->warning(FLERR,"Use special bonds = 0,1,1 with bond style sw");
  }
}

/* ---------------------------------------------------------------------- */

double BondStillingerWeber::equilibrium_distance(int i)
{
  return 0.5 * (lmin[i] + lmax[i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void BondStillingerWeber::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&lmin[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&lmax[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&delta[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&sigma[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void BondStillingerWeber::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR,&k[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&lmin[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&lmax[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&delta[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&sigma[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
  }
  MPI_Bcast(&k[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&lmin[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&lmax[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&delta[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&sigma[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondStillingerWeber::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g %g %g\n", i, k[i], lmin[i], lmax[i], delta[i], sigma[i]);
}

/* ---------------------------------------------------------------------- */

double BondStillingerWeber::single(int type, double rsq, int /*i*/, int /*j*/,
                        double &fforce)
{
  double a, b, c, u;

  double r = std::sqrt(rsq);
  double l0 = lmin[type];
  double l3 = lmax[type];
  double l1 = l0 + delta[type];
  double l2 = l3 - delta[type];

  fforce = 0.0;
  double eng = 0.0;

  if (r < l1) {
    if (r <= l0) {
        WARN("SW bond too short at step {}: {}", update->ntimestep, r);
        if (r <= 0.5 * l0) ERROR_ONE("Bad SW bond");
        else r = 0.95 * l0 + 0.05 * l1;
      }
    
    a = sigma[type] / (r - l0);
    b = sigma[type] / (r - l1);
    c = std::exp(b);
    u = k[type] * a * c;

    fforce += u * (b * b + a) / (sigma[type] * r);
    eng += u;
  }

  if (r > l2) {
    if (r >= l3) {
        WARN("SW bond too long at step {}: {}", update->ntimestep, r);
        if (r >= 2.0 * l3) ERROR_ONE("Bad SW bond");
        else r = 0.95 * l3 + 0.05 * l2;
      }
    
    a = sigma[type] / (l3 - r);
    b = sigma[type] / (l2 - r);
    c = std::exp(b);
    u = k[type] * a * c;

    fforce -= u * (b * b + a) / (sigma[type] * r);
    eng += u;
  }

  return eng;
}

/* ---------------------------------------------------------------------- */

void *BondStillingerWeber::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str,"k")==0) return (void*) k;
  if (strcmp(str,"lmin")==0) return (void*) lmin;
  if (strcmp(str,"lmax")==0) return (void*) lmax;
  if (strcmp(str,"delta")==0) return (void*) delta;
  if (strcmp(str,"sigma")==0) return (void*) sigma;
  return nullptr;
}
