/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
  -------------------------------------------------------------------------
  Contributed by Stefan Paquay @ Eindhoven University of Technology
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   This class implements the "Stillinger-Weber" potential used by Noguchi & Gompper, 2005, for non-bonded atoms:

   U_rep(r) = k*sigma*exp(sigma/(r-lmin-delta))/(r-lmin), r < lmin + delta
            = 0, r >= lmin + delta

   It is intended to be used with bond_style sw, which implements attractive and repulsive interactions between bonded atoms.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(sw_noguchi,PairStillingerWeber)

#else

#ifndef LMP_PAIR_STILLINGER_WEBER_H
#define LMP_PAIR_STILLINGER_WEBER_H

#include "pair.h"

namespace LAMMPS_NS {

class PairStillingerWeber : public Pair {
 public:
  PairStillingerWeber(class LAMMPS *);
  virtual ~PairStillingerWeber();

  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

 protected:
  double cut_global;
  double **cut;
  double **k;
  double **lmin;
  double **delta;
  double **sigma;

  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/
