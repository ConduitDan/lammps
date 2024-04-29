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

#include "bond_power.h"

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

BondPower::BondPower(LAMMPS *lmp)
  : Bond(lmp),
  k{nullptr},
  lmin{nullptr},
  lmax{nullptr},
  r{nullptr}
{}

/* ---------------------------------------------------------------------- */

BondPower::~BondPower()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(lmin);
    memory->destroy(lmax);
    memory->destroy(r);
  }
}

/* ---------------------------------------------------------------------- */

void BondPower::compute(int eflag, int vflag)
{
  int i1,i2,n,type;
  double u, lc0, lc1;
  double dx, dy, dz, r2, l,r_0;
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
    domain->minimum_image(dx, dy, dz);

    r2 = dx*dx + dy*dy + dz*dz;
    l = std::sqrt(r2);

    lc1 = lmin[type];
    lc0 = lmax[type];
    r_0 = r[type];
    
    if (l < lc1) {
      if (l <= 0.5 * lc1) {
        LOGWARN("Bond very short at step {}: {} ({}, {})",
          update->ntimestep, l, atom->tag[i1], atom->tag[i2]
        );
        if (l <= 0.25 * lc1) ERROR_ONE("Bad Power bond");
      }
      
      u = k[type]*std::pow(lc1-l,r_0)*std::pow(r_0,1+r_0);

      fbond += k[type]*std::pow(lc1-l,r_0-1)*std::pow(r_0,2+r_0);
      if (eflag) {
        ebond += u;
      }
    }
    if (l > lc0) {
      if (l >= 2 * lc0) {
        LOGWARN("Bond very long at step {}: {} ({}, {})",
          update->ntimestep, l, atom->tag[i1], atom->tag[i2]
        );
        if (l >= 4 * lc0) ERROR_ONE("Bad Power bond");
      }
      
      u = k[type]*std::pow(l-lc0,r_0)*std::pow(r_0,1+r_0);

      fbond -= k[type]*std::pow(l-lc0,r_0-1)*std::pow(r_0,2+r_0);
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

void BondPower::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(k,n+1,"bond:k");
  memory->create(lmin,n+1,"bond:lmin");
  memory->create(lmax,n+1,"bond:lmax");
  memory->create(r,n+1,"bond:r");
  memory->create(setflag,n+1,"bond:setflag");

  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

void BondPower::coeff(int narg, char **arg)
{
  if (narg != 5) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nbondtypes,ilo,ihi,error);

  double k_one = utils::numeric(FLERR,arg[1],false,lmp);
  double lmin_one = utils::numeric(FLERR,arg[2],false,lmp);
  double lmax_one = utils::numeric(FLERR,arg[3],false,lmp);
  double r_one = utils::numeric(FLERR,arg[4],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    lmin[i] = lmin_one;
    lmax[i] = lmax_one;
    r[i] = r_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   check if special_bond settings are valid
------------------------------------------------------------------------- */

void BondPower::init_style()
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

double BondPower::equilibrium_distance(int i)
{
  return 0.5 * (lmin[i] + lmax[i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void BondPower::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&lmin[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&lmax[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r[1],sizeof(int),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void BondPower::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR,&k[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&lmin[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&lmax[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&r[1],sizeof(int),atom->nbondtypes,fp,nullptr,error);
  }
  MPI_Bcast(&k[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&lmin[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&lmax[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r[1],atom->nbondtypes,MPI_INT,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondPower::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g %d\n", i, k[i], lmin[i], lmax[i], r[i]);
}

/* ---------------------------------------------------------------------- */

double BondPower::single(int type, double rsq, int /*i*/, int /*j*/,
                        double &fforce)
{
  double a, b, c, u;

  double l = std::sqrt(rsq);
  double lc1 = lmin[type];
  double lc0 = lmax[type];
  double r_0 = r[type];
  double eng = 0.0;
  fforce = 0;
    if (l < lc1) {
      if (l <= 0.5 * lc1) {
        LOGWARN("Bond very short at step {}: {}", update->ntimestep, l);
        if (l <= 0.25 * lc1) ERROR_ONE("Bad Power bond");
      }
      
      u = k[type]*std::pow(lc1-l,r_0)*std::pow(r_0,1+r_0);
      eng += u;
      fforce += k[type]*std::pow(lc1-l,r_0-1)*std::pow(r_0,2+r_0);
      
    }
    if (l > lc0) {
      if (l >= 2 * lc0) {
        LOGWARN("Bond very long at step {}: {}", update->ntimestep, l);
        if (l >= 4 * lc0) ERROR_ONE("Bad Power bond");
      }
      
      u = k[type]*std::pow(l-lc0,r_0)*std::pow(r_0,1+r_0);
      eng += u;
      fforce -= k[type]*std::pow(l-lc0,r_0-1)*std::pow(r_0,2+r_0);

    }

  return eng;
}

/* ---------------------------------------------------------------------- */

void *BondPower::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str,"k")==0) return (void*) k;
  if (strcmp(str,"lmin")==0) return (void*) lmin;
  if (strcmp(str,"lmax")==0) return (void*) lmax;
  if (strcmp(str,"r")==0) return (void*) r;
  return nullptr;
}
