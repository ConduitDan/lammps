/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(propel/bond, FixPropelBond)

#else

#ifndef LMP_FIX_ACTIVE_BOND_H
#define LMP_FIX_ACTIVE_BOND_H

#include "fix.h"

namespace LAMMPS_NS {

class FixPropelBond : public Fix {
  enum ReversalMode {
    NONE,
    PERIODIC,
    STOCHASTIC,
  };

 public:
  FixPropelBond(class LAMMPS *, int, char **);
  ~FixPropelBond();

  virtual void init();
  virtual int setmask();
  virtual void pre_force(int);
  virtual double memory_usage();

 private:
  double magnitude;
  double reversal_time;
  double reversal_prob;
  int nmolecules;
  ReversalMode mode;
  
  int *apply_to_type;
  int *reverse;
  class RanMars *random;

  void grow_reversal_list();
  void update_reversal_time();
};

} // namespace LAMMPS_NS

#endif // LMP_FIX_ACTIVE_BOND_H
#endif // FIX_CLASS