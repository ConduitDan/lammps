/**
 * compute_active_atom.h
 *
 * Provides the "compute active/atom" command in LAMMPS which computes active
 * forces along a polymer tangent. The forces can be polar or apolar (nematic),
 * with nematic forcing implemented through a stochastic reversing process.
 *
 * Usage:
 *  compute <id> <group> active/atom <style> <force> <args...>
 *
 * Arguments:
 *  id    : unique compute id/name for the computation
 *  group : group of atoms to perform computation on
 *  style : active forcing style ("polar" or "apolar" or "nematic")
 *          Note: apolar is equivalent to nematic
 *  force : magnitude of the applied active force
 *  args  : additional arguments for a particular style
 *          if style = apolar
 *              args... = No additional arguments
 *          if style = polar or nematic
 *              args... = <seed> <mean-reversal-time>
 *              seed = integer to seed random number generator (required)
 *              mean-reversal-time = timescale over which direction of active
 *                                   force reverses.
 *
 * Notes:
 *  This compute requires "newton off" to be set, due to certain assumptions
 *  about how bonds are stored. This might be fixed in a future version.
 *
 * Log:
 *  Created by: Abhijeet Joshi 
 *  Maintained by: Matthew Peterson <matthew.se.peterson@gmail.com>
 *
 *  2019/12/08
 *      - Some minor code cleaning
 *      - Changed compute name from "tgt/atom" to "active/atom" to better match
 *        the actual purpose
 *
 *  2019/07/19 
 *      - Fixed bug where polymers experience forces perpendicular to tangent
 *      - Fixed bug where polymers straddling processor domains could have
 *        inconsistent active force directions
 *
 *  2016/??/??
 *      - Created
 */


#ifdef COMPUTE_CLASS

ComputeStyle(active/atom,ComputeActiveAtom)

#else

#ifndef LMP_COMPUTE_ACTIVE_ATOM_H
#define LMP_COMPUTE_ACTIVE_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeActiveAtom : public Compute {
public:
    ComputeActiveAtom(class LAMMPS *, int, char **);
    ~ComputeActiveAtom();
    void init();
    void setup();
    void compute_peratom();
    double memory_usage();

private:
    int seed;
    int nmax;
    int nvalues;
    int ncount;
    int *bstyle;
    int singleflag;
    tagint maxmol;
    double fstr;
    double omega;
    double tau;
    double *randphase;
    double *poisson_delt; 
    double *next_time; 
    double **factive;
    double **array;
    class NeighList *list;
    class RanMars *randtgt;
};

}

#endif
#endif
