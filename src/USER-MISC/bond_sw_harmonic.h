/* -*- c++ -*- ----------------------------------------------------------
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
   This class implements the "Stillinger-Weber" potential used by Noguchi & Gompper, 2005:

   U_bond(r) = k*sigma*exp(sigma/(lmax-delta-r))/(lmax-r), r > lmax-delta
             = 0, r <= lmax-delta

   U_rep(r) = k*sigma*exp(sigma/(r-lmin-delta))/(r-lmin), r < lmin + delta
            = 0, r >= lmin + delta

   It is intended to be used with pair_style sw, which implements repulsion between non-bonded pairs of atoms.
------------------------------------------------------------------------- */

#ifdef BOND_CLASS

BondStyle(swHarmonic,BondStillingerWeberHarmonic)

#else

#ifndef LMP_BOND_STILLINGER_WEBER_HARMONIC_H
#define LMP_BOND_STILLINGER_WEBER_HARMONIC_H

#include "bond.h"

namespace LAMMPS_NS {

class BondStillingerWeberHarmonic : public Bond {
 public:
  BondStillingerWeberHarmonic(class LAMMPS *);
  virtual ~BondStillingerWeberHarmonic();
  virtual void compute(int, int);
  virtual void coeff(int, char **);
  void init_style();
  double equilibrium_distance(int);
  virtual void write_restart(FILE *);
  void read_restart(FILE *);
  void write_data(FILE *);
  double single(int, double, int, int, double &);
  virtual void *extract(const char *, int &);

 protected:
  double *k;
  double *k_harmonic;
  double *lmin;
  double *lmax;
  double *delta;
  double *sigma;


  virtual void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

W: FENE bond too long: %ld %d %d %g

A FENE bond has stretched dangerously far.  Its interaction strength
will be truncated to attempt to prevent the bond from blowing up.

E: Bad FENE bond

Two atoms in a FENE bond have become so far apart that the bond cannot
be computed.

E: Incorrect args for bond coefficients

Self-explanatory.  Check the input script or data file.

W: Use special bonds = 0,1,1 with bond style fene

Most FENE models need this setting for the special_bonds command.

W: FENE bond too long: %ld %g

A FENE bond has stretched dangerously far.  It's interaction strength
will be truncated to attempt to prevent the bond from blowing up.

*/
