/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
   Steve Plimpton, sjplimp@sandia.gov, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(ablate,FixAblate)

#else

#ifndef SPARTA_FIX_ABLATE_H
#define SPARTA_FIX_ABLATE_H

#include "fix.h"

namespace SPARTA_NS {

class FixAblate : public Fix {
 public:
  int igroup,dim;

  FixAblate(class SPARTA *, int, char **);
  ~FixAblate();
  int setmask();
  void init();
  void setup() {}
  void end_of_step();

  int pack_grid_one(int, char *, int);
  int unpack_grid_one(int, char *);
  void copy_grid_one(int, int);
  void reset_grid_count(int);
  void add_grid_one();
  double compute_scalar();
  double memory_usage();

  void store_corners(int, int, int, double *, double *,
                     double **, int *, double);

 protected:
  int me;
  int groupbit,which,argindex,icompute,ifix,maxrandom;
  double scale;
  char *idsource;
  int storeflag;
  int nx,ny,nz,ncorner;
  double thresh;
  double sum_delta;
   
  int nglocal;            // # of owned grid cells

  int **ixyz;             // ix,iy,iz indices (1 to Nxyz) of my cells
                          // in 2d/3d ablate grid (iz = 1 for 2d)

  double *celldelta;      // per-cell delta from compute or fix source
  double **cdelta;        // per-corner point deltas
  double **cdelta_ghost;  // ditto for my ghost cells communicated to me
  int maxgrid;            // max size of per-cell vectors/arrays
  int maxghost;           // max size of cdelta_ghost

  int *proclist;
  int *locallist;
  int *numsend;
  int maxsend;

  double *sbuf;
  int maxbuf;

  class MarchingSquares *ms;
  class MarchingCubes *mc;
  class RanPark *random;

  void set_delta_random();
  void set_delta();
  void decrement();
  void sync();
  int walk_to_neigh(int, int, int, int);
  void push2d() {}
  void push3d() {}
  void grow_percell(int);
  void grow_send();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

*/