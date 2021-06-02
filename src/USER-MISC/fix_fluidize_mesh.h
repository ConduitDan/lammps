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

FixStyle(fluidize/mesh, FixFluidizeMesh)

#else

#ifndef LMP_FIX_FLUIDIZE_MESH_H
#define LMP_FIX_FLUIDIZE_MESH_H

#include "fix.h"

namespace LAMMPS_NS {

class FixFluidizeMesh : public Fix {
  struct bond_type;
  struct dihedral_type;

public:
  FixFluidizeMesh(class LAMMPS *, int, char **);
  virtual ~FixFluidizeMesh();

  int setmask() override;
  void init() override;
  void post_integrate() override;

private:
  class RanMars *random;
  class Compute *temperature;
  dihedral_type *staged_swaps;
  dihedral_type *staged_swaps_all;
  int num_staged_swaps;
  int max_staged_swaps;
  int num_staged_swaps_all;
  int max_staged_swaps_all;
  double swap_probability;
  double rmax2;
  double rmin2;
  double kbt;

  void stage_swap(dihedral_type);
  void gather_swaps();
  void commit();

  bool accept_change(bond_type, bond_type);
  void try_swap(int, int, int, int);

  int find_bond(bond_type &);
  void remove_bond_at(bond_type, int);
  void insert_bond(bond_type);

  int find_dihedral(dihedral_type &);
  void remove_dihedral_at(dihedral_type, int);
  void insert_dihedral(dihedral_type);
  void swap_dihedrals(dihedral_type, dihedral_type);

  bool is_owned(dihedral_type);
  int find_exterior_atom(dihedral_type, dihedral_type);
};

} // namespace LAMMPS_NS

#endif
#endif

/*
   ERROR/WARNING messages:

   E: Illegal ... command

   Self-explanatory.  Check the input script syntax and compare to the
   documentation for the command.  You can use -echo screen as a
   command-line option when running LAMMPS to see the offending line.

*/
