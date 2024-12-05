/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.github.io
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "collide_vss_kokkos.h"
#include "grid.h"
#include "update.h"
#include "particle_kokkos.h"
#include "mixture.h"
#include "collide.h"
#include "react.h"
#include "comm.h"
#include "random_knuth.h"
#include "random_mars.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "error.h"
#include "kokkos.h"
#include "sparta_masks.h"
#include "modify.h"
#include "fix.h"
#include "fix_ambipolar.h"

using namespace SPARTA_NS;
using namespace MathConst;

enum{NONE,DISCRETE,SMOOTH};            // several files
enum{CONSTANT,VARIABLE};

#define DELTAGRID 1000            // must be bigger than split cells per cell
#define DELTADELETE 1024
#define DELTAELECTRON 128
#define DELTACELLCOUNT 2

#define MAXLINE 1024
#define BIG 1.0e20


/* ----------------------------------------------------------------------
   NTC algorithm for a single group
------------------------------------------------------------------------- */

void CollideVSSKokkos::collisions_one_sw(COLLIDE_REDUCE &reduce)
{
  // loop over cells I own

  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK);
  if (vibstyle == DISCRETE) particle_kk->sync(Device,CUSTOM_MASK);
  d_particles = particle_kk->k_particles.d_view;
  d_species = particle_kk->k_species.d_view;
  d_ewhich = particle_kk->k_ewhich.d_view;
  k_eiarray = particle_kk->k_eiarray;

  GridKokkos* grid_kk = (GridKokkos*) grid;
  grid_kk->sync(Device,CINFO_MASK);
  d_plist = grid_kk->d_plist;


  grid_kk_copy.copy(grid_kk);

  if (sparta->kokkos->atomic_reduction) {
    if (sparta->kokkos->need_atomics)
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCollideCollisionsOneSW<1> >(0,nglocal),*this);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCollideCollisionsOneSW<0> >(0,nglocal),*this);
  } else
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagCollideCollisionsOneSW<-1> >(0,nglocal),*this,reduce);
}

template < int ATOMIC_REDUCTION >
KOKKOS_INLINE_FUNCTION
void CollideVSSKokkos::operator()(TagCollideCollisionsOneSW< ATOMIC_REDUCTION >, const int &icell) const {
  COLLIDE_REDUCE reduce;
  this->template operator()< ATOMIC_REDUCTION >(TagCollideCollisionsOneSW< ATOMIC_REDUCTION >(), icell, reduce);
}

template < int ATOMIC_REDUCTION >
KOKKOS_INLINE_FUNCTION
void CollideVSSKokkos::operator()(TagCollideCollisionsOneSW< ATOMIC_REDUCTION >, const int &icell, COLLIDE_REDUCE &reduce) const {
  if (d_retry()) return;

  int np = grid_kk_copy.obj.d_cellcount[icell];
  if (np <= 1) return;

  const double volume = grid_kk_copy.obj.k_cinfo.d_view[icell].volume / grid_kk_copy.obj.k_cinfo.d_view[icell].weight;
  if (volume == 0.0) d_error_flag() = 1;

  // find max particle weight in this cell
  double swgt_max = 0.0;
  for (int n = 0; n < np; n++)
    if (swpmflag)
      swgt_max = MAX(d_particles[d_plist(icell,n)].weight, swgt_max);
  swgt_max *= fnum;
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double CollideVSSKokkos::attempt_collision_sw_kokkos(int icell, int np, double volume, double pre_wtf,
                                                     double sweight_max, rand_type &rand_gen) const
{
 double nattempt;
 double fnum_local = sweight_max*(1.0+pre_wtf*wtf);

 if (remainflag) {
   nattempt = 0.5 * np * (np-1) *
     d_vremax(icell,0,0) * dt * fnum_local / volume + d_remain(icell,0,0);
   d_remain(icell,0,0) = nattempt - static_cast<int> (nattempt);
 } else {
   nattempt = 0.5 * np * (np-1) *
     d_vremax(icell,0,0) * dt * fnum_local / volume + rand_gen.drand();
 }

  return nattempt;
}

/* ----------------------------------------------------------------------
   determine if collision actually occurs
   1 = yes, 0 = no
   update vremax either way
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int CollideVSSKokkos::test_collision_sw_kokkos(int icell, int igroup, int jgroup,
                                               Particle::OnePart *ip, Particle::OnePart *jp,
                                               struct State &precoln, double sweight_max,
                                               rand_type &rand_gen) const
{
  double *vi = ip->v;
  double *vj = jp->v;
  int ispecies = ip->ispecies;
  int jspecies = jp->ispecies;
  double du  = vi[0] - vj[0];
  double dv  = vi[1] - vj[1];
  double dw  = vi[2] - vj[2];
  double vr2 = du*du + dv*dv + dw*dw;
  double vro  = pow(vr2,1.0-d_params(ispecies,jspecies).omega);

  double ijsw = 1.0;
  double isw = ip->weight*fnum;
  double jsw = jp->weight*fnum;
  ijsw = MAX(isw ,jsw)/sweight_max;

  // although the vremax is calcualted for the group,
  // the individual collisions calculated species dependent vre

  double vre = vro*d_prefactor(ispecies,jspecies);
  d_vremax(icell,igroup,jgroup) = MAX(vre,d_vremax(icell,igroup,jgroup));
  if (vre/d_vremax(icell,igroup,jgroup)*ijsw < rand_gen.drand()) return 0;
  precoln.vr2 = vr2;
  return 1;
}
