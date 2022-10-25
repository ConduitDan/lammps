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

#include "pair_sw.h"

#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "update.h"
#include "force.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

#define ERROR_ALL(...) error->all(FLERR, fmt::format(__VA_ARGS__))
#define ERROR_ONE(...) error->one(FLERR, fmt::format(__VA_ARGS__))
#define WARN(...) error->warning(FLERR, fmt::format(__VA_ARGS__), 0)
#define LOGWARN(...) error->warning(FLERR, fmt::format(__VA_ARGS__), 1)

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairStillingerWeber::PairStillingerWeber(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairStillingerWeber::~PairStillingerWeber()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cut);

    memory->destroy(k);
    memory->destroy(lmin);
    memory->destroy(delta);
    memory->destroy(sigma);
  }
}

/* ---------------------------------------------------------------------- */

void PairStillingerWeber::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double a, b, c, u, l0, l1, l2, l3;
  double rsq,r,dr,dexp,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      r = sqrt(rsq);
      jtype = type[j];
      l0 = lmin[itype][jtype];
      l1 = l0 + delta[itype][jtype];

      if (r < l1) {
        if (r<= l0) {
          LOGWARN("SW non-bonded pair distance too short at step {}: {} ({}, {})",
          update->ntimestep, r, atom->tag[i], atom->tag[j]
          );
          if (r <= 0.5 * l0) ERROR_ONE("Bad SW non-bonded pair");
          else r = 0.95 * l0 + 0.05 * l1;  
        }

        a = sigma[itype][jtype] / (r - l0);
        b = sigma[itype][jtype] / (r - l1);
        c = std::exp(b);
        u = k[itype][jtype] * a * c;

        fpair = factor_lj * u * (b * b + a) / (sigma[itype][jtype] * r);

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = u;
          evdwl *= factor_lj;
        }
        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairStillingerWeber::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(k,n+1,n+1,"pair:k");
  memory->create(lmin,n+1,n+1,"pair:lmin");
  memory->create(delta,n+1,n+1,"pair:delta");
  memory->create(sigma,n+1,n+1,"pair:sigma");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairStillingerWeber::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = utils::numeric(FLERR,arg[0],false,lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairStillingerWeber::coeff(int narg, char **arg)
{
  if (narg < 5 || narg > 6) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double k_one = utils::numeric(FLERR,arg[2],false,lmp);
  double lmin_one = utils::numeric(FLERR,arg[3],false,lmp);
  double delta_one = utils::numeric(FLERR,arg[4],false,lmp);
  double sigma_one = 1.0;
  if (narg == 5) sigma_one = utils::numeric(FLERR,arg[5],false,lmp);

  double cut_one = cut_global;

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      k[i][j] = k_one;
      lmin[i][j] = lmin_one;
      delta[i][j] = delta_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}


/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairStillingerWeber::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  k[j][i] = k[i][j];
  lmin[j][i] = lmin[i][j];
  delta[j][i] = delta[i][j];
  sigma[j][i] = sigma[i][j];
  cut[j][i] = cut[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairStillingerWeber::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&k[i][j],sizeof(double),1,fp);
        fwrite(&lmin[i][j],sizeof(double),1,fp);
        fwrite(&delta[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairStillingerWeber::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&k[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&lmin[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&delta[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&sigma[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cut[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&k[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&lmin[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&delta[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairStillingerWeber::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairStillingerWeber::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&cut_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairStillingerWeber::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %g\n",i,k[i][i],lmin[i][i],delta[i][i],sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairStillingerWeber::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %d\n",
              i,j,k[i][j],lmin[i][j],delta[i][j],sigma[i][j], cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairStillingerWeber::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                                     double /*factor_coul*/, double factor_lj,
                                     double &fforce)
{
  double a, b, c, u;

  double r = std::sqrt(rsq);
  double l0 = lmin[itype][jtype];
  double l1 = l0 + delta[itype][jtype];

  fforce = 0.0;
  double eng = 0.0;

  if (r < l1) {
    if (r <= l0) {
        WARN("SW bond too short at step {}: {}", update->ntimestep, r);
        if (r <= 0.5 * l0) ERROR_ONE("Bad SW bond");
        else r = 0.95 * l0 + 0.05 * l1;
      }
    
    a = sigma[itype][jtype] / (r - l0);
    b = sigma[itype][jtype] / (r - l1);
    c = std::exp(b);
    u = k[itype][jtype] * a * c;

    fforce += u * (b * b + a) / (sigma[itype][jtype] * r);
    eng += u;
  }

  return eng;
}

/* ---------------------------------------------------------------------- */

void *PairStillingerWeber::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"k") == 0) return (void *) k;
  if (strcmp(str,"lmin") == 0) return (void *) lmin;
  if (strcmp(str,"delta") == 0) return (void *) delta;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  return nullptr;
}
