#ifdef FIX_CLASS

FixStyle(bond/flip, FixBondFlip)

#else

#ifndef LMP_FIX_BOND_FLIP_H
#define LMP_FIX_BOND_FLIP_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBondFlip : public Fix {
public:
    FixBondFlip(LAMMPS *, int, char **);
    ~FixBondFlip();
    int setmask();
    void init();
    void end_of_step();

private:
    void bcast_atom(int &, const int) const;
    double compute_dihedral_energy(const int, const int, const int, const int) const;
    bool find_dihedral(const int, const int, int &, int &) const;
    bool find_dihedral_other(const int, const int, int &, int &) const;
    void delete_bond(const int, const int) const;
    void create_bond(const int, const int) const;
    void delete_dihedral(const int, const int) const;
    void create_dihedral(const int, const int, const int, const int) const;
    void pick() const;
    bool flip() const;

    bool consider_energy_change;
    int r_every;
    int num_attempts;
    double blmin;
    double blmax;
    int btype;
    int dtype;
    double k;
    int seed;

    mutable int bproc;
    mutable int bm;
    mutable int bn;

    mutable int r_flipped;
    mutable int r_attempts;

    class RanMars *random_all;
    class RanMars *random_one;
};

}

#endif
#endif
