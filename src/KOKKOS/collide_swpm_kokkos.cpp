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
enum{ENERGY,HEAT,STRESS}; // particle reduction choices
enum{BINARY,WEIGHT}; // grouping choices

#define DELTAGRID 1000            // must be bigger than split cells per cell
#define DELTADELETE 1024
#define DELTAELECTRON 128
#define DELTACELLCOUNT 2

#define MAXLINE 1024
#define BIG 1.0e20


/* ----------------------------------------------------------------------
   Stochastic weighted algorithm
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

  copymode = 1;

  /* ATOMIC_REDUCTION: 1 = use atomics
                       0 = don't need atomics
                      -1 = use parallel_reduce
  */

  // Stochastic weighting may create or delete more particles than existing views can hold.
  //  Cannot grow a Kokkos view in a parallel loop, so
  //  if the capacity of the view is exceeded, break out of parallel loop,
  //  reallocate on the host, and then repeat the parallel loop again.
  //  Unfortunately this leads to really messy code.

  h_retry() = 1;

  double extra_factor = 1.1; // should we define a swpm_retry_flag?
  auto maxdelete_extra = maxdelete*extra_factor;
  if (d_dellist.extent(0) < maxdelete_extra) {
    memoryKK->destroy_kokkos(k_dellist,dellist);
    memoryKK->create_kokkos(k_dellist,dellist,maxdelete_extra,"collide:dellist");
    d_dellist = k_dellist.d_view;
  }

  maxcellcount = particle_kk->get_maxcellcount();
  auto maxcellcount_extra = maxcellcount*extra_factor;
  if (d_plist.extent(1) < maxcellcount_extra) {
    d_plist = {};
    Kokkos::resize(grid_kk->d_plist,nglocal,maxcellcount_extra);
    d_plist = grid_kk->d_plist;
  }

  auto nlocal_extra = particle->nlocal*extra_factor;
  if (d_particles.extent(0) < nlocal_extra) {
    particle->grow(nlocal_extra - particle->nlocal);
    d_particles = particle_kk->k_particles.d_view;
    k_eiarray = particle_kk->k_eiarray;
  }

  while (h_retry()) {

    backup(); // conditional if we introduced a swpm_retry_flag

    h_retry() = 0;
    h_maxdelete() = maxdelete;
    h_maxcellcount() = maxcellcount;
    h_part_grow() = 0;
    h_ndelete() = 0;
    h_nlocal() = particle->nlocal;

    Kokkos::deep_copy(d_scalars,h_scalars);

    grid_kk_copy.copy(grid_kk);

    if (sparta->kokkos->atomic_reduction) {
      if (sparta->kokkos->need_atomics)
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCollideCollisionsOneSW<1> >(0,nglocal),*this);
      else
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCollideCollisionsOneSW<0> >(0,nglocal),*this);
    } else
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagCollideCollisionsOneSW<-1> >(0,nglocal),*this,reduce);

    Kokkos::deep_copy(h_scalars,d_scalars);

    if (h_retry()) {
      //printf("Retrying, reason %i %i %i !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",h_maxdelete() > d_dellist.extent(0),h_maxcellcount() > d_plist.extent(1),h_part_grow());

      restore(); // conditional if we introduced a swpm_retry_flag (see collisions_one() logic for reactions)

      reduce = COLLIDE_REDUCE();

      maxdelete = h_maxdelete();
      if (d_dellist.extent(0) < maxdelete) {
        memoryKK->destroy_kokkos(k_dellist,dellist);
        memoryKK->grow_kokkos(k_dellist,dellist,maxdelete,"collide:dellist");
        d_dellist = k_dellist.d_view;
      }

      maxcellcount = h_maxcellcount();
      particle_kk->set_maxcellcount(maxcellcount);
      if (d_plist.extent(1) < maxcellcount) {
        d_plist = {};
        Kokkos::resize(grid_kk->d_plist,nglocal,maxcellcount);
        d_plist = grid_kk->d_plist;
      }

      auto nlocal_new = h_nlocal();
      if (d_particles.extent(0) < nlocal_new) {
        particle->grow(nlocal_new - particle->nlocal);
        d_particles = particle_kk->k_particles.d_view;
        k_eiarray = particle_kk->k_eiarray;
      }
    }
  }

  ndelete = h_ndelete();

  particle->nlocal = h_nlocal();

  copymode = 0;

  if (h_error_flag())
    error->one(FLERR,"Collision cell volume is zero");

  particle_kk->modify(Device,PARTICLE_MASK);

  d_particles = t_particle_1d(); // destroy reference to reduce memory use
  d_plist = {};
}

void CollideVSSKokkos::group_reduce(COLLIDE_REDUCE &reduce)
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
  d_pL = grid_kk->d_pL;
  d_pLU = grid_kk->d_pLU;

  boltz = update->boltz;

  copymode = 1;

  /* ATOMIC_REDUCTION: 1 = use atomics
                       0 = don't need atomics
                      -1 = use parallel_reduce
  */

  // Stochastic weighting particle reduction may delete more particles than existing views can hold.
  //  Cannot grow a Kokkos view in a parallel loop, so
  //  if the capacity of the view is exceeded, break out of parallel loop,
  //  reallocate on the host, and then repeat the parallel loop again.
  //  Unfortunately this leads to really messy code.

  h_retry() = 1;

  double extra_factor = 1.1; // should we define a swpm_retry_flag?
  auto maxdelete_extra = maxdelete*extra_factor;
  if (d_dellist.extent(0) < maxdelete_extra) {
    memoryKK->destroy_kokkos(k_dellist,dellist);
    memoryKK->create_kokkos(k_dellist,dellist,maxdelete_extra,"collide:dellist");
    d_dellist = k_dellist.d_view;
  }

  maxcellcount = particle_kk->get_maxcellcount();
  auto maxcellcount_extra = maxcellcount*extra_factor;
  if (d_plist.extent(1) < maxcellcount_extra) {
    d_plist = {};
    Kokkos::resize(grid_kk->d_plist,nglocal,maxcellcount_extra);
    d_plist = grid_kk->d_plist;
  }

  auto nlocal_extra = particle->nlocal*extra_factor;
  if (d_particles.extent(0) < nlocal_extra) {
    particle->grow(nlocal_extra - particle->nlocal);
    d_particles = particle_kk->k_particles.d_view;
    k_eiarray = particle_kk->k_eiarray;
  }

  while (h_retry()) {

    backup(); // conditional if we introduced a swpm_retry_flag

    h_retry() = 0;
    h_maxdelete() = maxdelete;
    h_maxcellcount() = maxcellcount;
    h_part_grow() = 0;
    h_ndelete() = 0;
    h_nlocal() = particle->nlocal;

    Kokkos::deep_copy(d_scalars,h_scalars);

    grid_kk_copy.copy(grid_kk);

    if (sparta->kokkos->atomic_reduction) {
      if (sparta->kokkos->need_atomics)
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCollideGroupReduce<1> >(0,nglocal),*this);
      else
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCollideGroupReduce<0> >(0,nglocal),*this);
    } else
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagCollideGroupReduce<-1> >(0,nglocal),*this,reduce);

    Kokkos::deep_copy(h_scalars,d_scalars);

    if (h_retry()) {
      //printf("Retrying, reason %i %i %i !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",h_maxdelete() > d_dellist.extent(0),h_maxcellcount() > d_plist.extent(1),h_part_grow());

      restore(); // conditional if we introduced a swpm_retry_flag (see collisions_one() logic for reactions)

      reduce = COLLIDE_REDUCE();

      maxdelete = h_maxdelete();
      if (d_dellist.extent(0) < maxdelete) {
        memoryKK->destroy_kokkos(k_dellist,dellist);
        memoryKK->grow_kokkos(k_dellist,dellist,maxdelete,"collide:dellist");
        d_dellist = k_dellist.d_view;
      }

      maxcellcount = h_maxcellcount();
      particle_kk->set_maxcellcount(maxcellcount);
      if (d_plist.extent(1) < maxcellcount) {
        d_plist = {};
        Kokkos::resize(grid_kk->d_plist,nglocal,maxcellcount);
        d_plist = grid_kk->d_plist;
      }

      auto nlocal_new = h_nlocal();
      if (d_particles.extent(0) < nlocal_new) {
        particle->grow(nlocal_new - particle->nlocal);
        d_particles = particle_kk->k_particles.d_view;
        k_eiarray = particle_kk->k_eiarray;
      }
    }
  }

  ndelete = h_ndelete();

  particle->nlocal = h_nlocal();

  copymode = 0;

  if (h_error_flag())
    error->one(FLERR,"Collision cell volume is zero");

  particle_kk->modify(Device,PARTICLE_MASK);

  d_particles = t_particle_1d(); // destroy reference to reduce memory use
  d_plist = {};
  d_pL = {};
  d_pLU = {};
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
  // NOTE:  deal with other things causing errors:  AKS

  struct State precoln;       // state before collision
  struct State postcoln;      // state after collision

  rand_type rand_gen = rand_pool.get_state();

  // find max particle weight in this cell
  double sweight_max_kk = 0.0;
  for (int n = 0; n < np; n++)
    if (swpmflag)
      sweight_max_kk = MAX(d_particles[d_plist(icell,n)].weight, sweight_max_kk);
  sweight_max_kk *= fnum;

  // attempt = exact collision attempt count for all particles in cell
  // nattempt = rounded attempt with RN
  // if no attempts, continue to next grid cell

  double pre_wtf_kk = 1.0;
  if (np >= Ncmin && Ncmin > 0.0)
    pre_wtf_kk = 0.0;

  const double attempt = attempt_collision_sw_kokkos(icell,np,volume,pre_wtf_kk,sweight_max_kk,rand_gen);
  const int nattempt = static_cast<int> (attempt);
  if (!nattempt){
    rand_pool.free_state(rand_gen);
    return;
  }
  if (ATOMIC_REDUCTION == 1)
    Kokkos::atomic_add(&d_nattempt_one(),nattempt);
  else if (ATOMIC_REDUCTION == 0)
    d_nattempt_one() += nattempt;
  else
    reduce.nattempt_one += nattempt;

  // perform collisions
  // select random pair of particles, cannot be same
  // test if collision actually occurs

  for (int m = 0; m < nattempt; m++) {
    const int i = np * rand_gen.drand();
    int j = np * rand_gen.drand();
    while (i == j) j = np * rand_gen.drand();

    Particle::OnePart* ipart = &d_particles[d_plist(icell,i)];
    Particle::OnePart* jpart = &d_particles[d_plist(icell,j)];
    Particle::OnePart* kpart;
    Particle::OnePart* lpart;
    Particle::OnePart* mpart;

    // test if collision actually occurs, then perform it
    if (!test_collision_sw_kokkos(icell,0,0,ipart,jpart,precoln,sweight_max_kk,rand_gen)) continue;

    // split particles
    int index_kpart, index_lpart;
    int newp = split(pre_wtf_kk,index_kpart,index_lpart,rand_gen,ipart,jpart,kpart,lpart);

    // add new particles to particle list
    if (newp > 0) {
      if (np+newp <= d_plist.extent(1)) { // check if new particles can be added without reallocation
        if (newp == 2) {
          d_plist(icell,np++) = index_kpart;
          d_plist(icell,np++) = index_lpart;
        }
        else if (newp == 1) {
          d_plist(icell,np++) = index_kpart;
        }
      }
      else {
        d_retry() = 1;
        d_maxcellcount() += DELTACELLCOUNT;
        rand_pool.free_state(rand_gen);
        return;
      }
    }

    //   since ipart and jpart have same weight, do not need
    //   ... to account for weight during collision itself
    //   also the splits are all handled beforehand
    mpart = NULL; // dummy particle
    int index_mpart;
    Particle::OnePart* recomb_part3 = NULL;
    int recomb_species = -1;
    double recomb_density = 0.0;
    setup_collision_kokkos(ipart,jpart,precoln,postcoln);
    perform_collision_kokkos(ipart,jpart,mpart,precoln,postcoln,rand_gen,
                             recomb_part3,recomb_species,recomb_density,index_mpart);

    if (ATOMIC_REDUCTION == 1)
      Kokkos::atomic_increment(&d_ncollide_one());
    else if (ATOMIC_REDUCTION == 0)
      d_ncollide_one()++;
    else
      reduce.ncollide_one++;
  }

  // mark particles with tiny weights for deletion
  double sw_mean = 0.0;
  int count = 0;
  for (int n = 0; n < np; n++) {
    double isw = d_particles[d_plist(icell,n)].weight;
    if (isw > 0.) sw_mean += isw;
  }
  sw_mean /= np;

  for (int n = 0; n < np; n++) {
    double isw = d_particles[d_plist(icell,n)].weight;
    if (isw < sw_mean*1e-5) {
      int ndelete = Kokkos::atomic_fetch_add(&d_ndelete(),1);
      if (ndelete < d_dellist.extent(0)) {
        d_dellist(ndelete) = d_plist(icell,n);
        d_particles[d_plist(icell, n)].weight = -1.0;
      } else {
        d_retry() = 1;
        d_maxdelete() += DELTADELETE;
        rand_pool.free_state(rand_gen);
        return;
      }
    }
  }

  rand_pool.free_state(rand_gen);
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

KOKKOS_INLINE_FUNCTION
int CollideVSSKokkos::split(double pre_wtf_kk,
                            int &index_kpart,
                            int &index_lpart,
                            rand_type &rand_gen,
                            Particle::OnePart *&ip,
                            Particle::OnePart *&jp,
                            Particle::OnePart *&kp,
                            Particle::OnePart *&lp)  const {
  double xk[3],vk[3];
  double xl[3],vl[3];
  double erotk, erotl;

  // checks if particles properly deleted
  kp = NULL;
  lp = NULL;

  // weight transfer function is assumed to be
  // ... MIN(ip->sweight,jp->sweight)/(1 + pre_wtf * wtf)

  double isw = ip->weight;
  double jsw = jp->weight;

  // error check goes here-----------
  if (isw <= 0.0 || jsw <= 0.0)
    ;
  // error check --------------------

  int ks, ls, kcell, lcell;
  double Gwtf, ksw, lsw;

  if(isw >= jsw) { // particle ip has larger weight
    Gwtf = jsw/(1.0+pre_wtf_kk*wtf);
    ksw  = isw-Gwtf;
    lsw  = jsw-Gwtf;

    ks = ip->ispecies;
    ls = jp->ispecies;

    kcell = ip->icell;
    lcell = jp->icell;

    memcpy(xk,ip->x,3*sizeof(double));
    memcpy(vk,ip->v,3*sizeof(double));
    memcpy(xl,jp->x,3*sizeof(double));
    memcpy(vl,jp->v,3*sizeof(double));

    erotk = ip->erot;
    erotl = jp->erot;

  } else {   // particle jp has larger weight
    Gwtf = isw/(1.0+pre_wtf_kk*wtf);
    ksw  = jsw-Gwtf;
    lsw  = isw-Gwtf;

    ks = jp->ispecies;
    ls = ip->ispecies;

    kcell = jp->icell;
    lcell = ip->icell;

    memcpy(xk,jp->x,3*sizeof(double));
    memcpy(vk,jp->v,3*sizeof(double));
    memcpy(xl,ip->x,3*sizeof(double));
    memcpy(vl,ip->v,3*sizeof(double));

    erotk = jp->erot;
    erotl = ip->erot;
  }

  // update weights

  ip->weight = Gwtf;
  jp->weight = Gwtf;

  // Gwtf should never be negative or zero
  if (Gwtf <= 0.0)
    ; // error->one(FLERR,"Negative weight assigned after split");
  if (Gwtf > 0.0 && pre_wtf_kk > 0.0)
    if (ksw <= 0.0 || lsw <= 0.0)
      ; // error->one(FLERR,"Zero or negative weight after split");

  // number of new particles
  int newp = 0;

  // gk is always the bigger of the two

  if(ksw > 0) {
    int id = MAXSMALLINT*rand_gen.drand();;
    index_kpart = Kokkos::atomic_fetch_add(&d_nlocal(),1);
    int reallocflag =
      ParticleKokkos::add_particle_kokkos(d_particles,index_kpart,id,ks,kcell,xk,vk,erotk,0.0);
    if (reallocflag) {
      d_retry() = 1;
      d_part_grow() = 1;
      return 0;
    }
    newp++;
    kp = &d_particles[index_kpart];
    kp->weight = ksw;
  }

  if (kp) {
    if (kp->weight <= 0.0) {
      ; // error output here
    }
  }

  // there should never be case where you add particle "l" if
  // ... you did not add particle "k"

  if(lsw > 0) {
    if(ksw <= 0)
      ; // error->one(FLERR,"Bad addition to particle list");

    int id = MAXSMALLINT*rand_gen.drand();;
    index_lpart = Kokkos::atomic_fetch_add(&d_nlocal(),1);
    int reallocflag =
      ParticleKokkos::add_particle_kokkos(d_particles,index_lpart,id,ls,lcell,xl,vl,erotl,0.0);
    if (reallocflag) {
      d_retry() = 1;
      d_part_grow() = 1;
      return 0;
    }
    newp++;
    lp = &d_particles[index_lpart];
    lp->weight = lsw;
  }

  if (lp) {
    if (lp->weight <= 0.0) {
      ; // error->one(FLERR,"New particle [l] has bad weight");
    }
  }
  return newp;
}

template < int ATOMIC_REDUCTION >
KOKKOS_INLINE_FUNCTION
void CollideVSSKokkos::operator()(TagCollideGroupReduce< ATOMIC_REDUCTION >, const int &icell) const {
  COLLIDE_REDUCE reduce;
  this->template operator()< ATOMIC_REDUCTION >(TagCollideGroupReduce< ATOMIC_REDUCTION >(), icell, reduce);
}

template < int ATOMIC_REDUCTION >
KOKKOS_INLINE_FUNCTION
void CollideVSSKokkos::operator()(TagCollideGroupReduce< ATOMIC_REDUCTION >, const int &icell, COLLIDE_REDUCE &reduce) const {
  if (d_retry()) return;

  int np = grid_kk_copy.obj.d_cellcount[icell];
  if (np <= Ncmax) return;

  int n = 0;
  for (int num = 0; num < np; num++) {
    double isw = d_particles[d_plist(icell,num)].weight;
    if (isw > 0.)
      n++;
  }

  int nold;
  double gbuf_kk = 0;
  while (n > Ncmax) {
    nold = n;
    if (group_type == BINARY) {
      group_bt_kokkos(n, gbuf_kk, icell);
    } else if (group_type == WEIGHT) {
      ;
    }
  }
}

KOKKOS_INLINE_FUNCTION
void CollideVSSKokkos::group_bt_kokkos(int np, double gbuf_kk, int icell) const {

  // ignore groups which have too few particles
  if (np <= Ngmin) return;

  // compute stress tensor since it's needed for
  // .. further dividing and reduction
  double gsum, msum, mV[3], mVV[3][3], mVVV[3][3];
  gsum = msum = 0.0;
  for (int i = 0; i < 3; i++) {
    mV[i] = 0.0;
    for (int j = 0; j < 3; j++) {
      mVV[i][j] = 0.0;
      mVVV[i][j] = 0.0;
    }
  }

  // find maximum particle weight
  int ispecies;
  double mass, psw, pmsw, vp[3];
  double Erot;
  for (int p = 0; p < np; p++) {
    ispecies = d_particles[d_plist(icell,p)].ispecies;
    mass = d_species[ispecies].mass;
    psw = d_particles[d_plist(icell,p)].weight;
    pmsw = psw * mass;
    memcpy(vp, d_particles[d_plist(icell,p)].v, 3*sizeof(double));
    gsum += psw;
    msum += pmsw;
    Erot += psw*d_particles[d_plist(icell,p)].erot;
    for (int i = 0; i < 3; i++) {
      mV[i] += (pmsw*vp[i]);
      for (int j = 0; j < 3; j++) {
        mVV[i][j] += (pmsw*vp[i]*vp[j]);
        mVVV[i][j] += (pmsw*vp[i]*vp[j]*vp[j]);
      }
    }
  }

  // mean velocity

  double V[3];
  for (int i = 0; i < 3; i++) V[i] = mV[i]/msum;

  // stress tensor

  double pij[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      pij[i][j] = mVV[i][j] - mV[i]*mV[j]/msum;

  // if group is small enough, merge the particles

  if (np <= Ngmax+gbuf_kk) {

    // temperature
    double T = (pij[0][0] + pij[1][1] + pij[2][2])/
      (3.0 * gsum * update->boltz);

    // heat flux
    double Vsq = V[0]*V[0] + V[1]*V[1] + V[2]*V[2];
    double h,h1,h2,q[3];
    int i1,i2;
    for (int i = 0; i < 3; i++) {
      if (i == 0) {
        i1 = 1;
        i2 = 2;
      } else if (i == 1) {
        i1 = 2;
        i2 = 0;
      } else {
        i1 = 0;
        i2 = 1;
      }

      h  = mVVV[i][i] - 3.0*mV[i]*mVV[i][i]/msum +
           2.0*mV[i]*mV[i]*mV[i]/msum/msum;
      h1 = mVVV[i][i1] - 2.0*mVV[i][i1]*mV[i1]/msum -
           mV[i]*mVV[i1][i1]/msum + 2.0*mV[i]*mV[i1]*mV[i1]/msum/msum;
      h2 = mVVV[i][i2] - 2.0*mVV[i][i2]*mV[i2]/msum -
           mV[i]*mVV[i2][i2]/msum + 2.0*mV[i]*mV[i2]*mV[i2]/msum/msum;
      q[i] = (h + h1 + h2) * 0.5;
    }

    // scale values to be consistent with definitions in
    // .. stochastic numerics book

    T *= boltz/mass;
    for (int i = 0; i < 3; i++) {
      q[i] /= mass;
      for(int j = 0; j < 3; j++) pij[i][j] /= mass;
    }
#if 0
    // reduce based on type
    if (reduction_type == ENERGY) {
      reduce(plist_leaf, np, gsum, V, T, Erot);
    } else if (reduction_type == HEAT) {
      reduce(plist_leaf, np, gsum, V, T, Erot, q);
    } else if (reduction_type == STRESS) {
      reduce(plist_leaf, np, gsum, V, T, Erot, q, pij);
    }
#endif
  // group still too large so divide further

  } else {

    // Compute covariance matrix

    double Rij[3][3];
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        Rij[i][j] = pij[i][j]/gsum;

    // Find eigenpairs

    double eval[3], evec[3][3];
    //    int ierror = MathEigen::jacobi3(Rij,eval,evec);

    // Find largest eigenpair

    double maxeval;
    double maxevec[3]; // normal of splitting plane

    maxeval = 0;
    for (int i = 0; i < 3; i++) {
      if (fabs(eval[i]) > maxeval) {
        maxeval = std::abs(eval[i]);
        for (int j = 0; j < 3; j++) {
          maxevec[j] = evec[j][i];
        }
      }
    }
    // Separate based on particle velocity

    double center = V[0]*maxevec[0] + V[1]*maxevec[1] + V[2]*maxevec[2];
    int pid, pidL[np], pidR[np];
    int npL, npR;
    npL = npR = 0;
#if 0
    for (int i = 0; i < np; i++) {
      pid = plist_leaf[i];
      ipart = &particles[pid];
      if (MathExtra::dot3(ipart->v,maxevec) < center)
        pidL[npL++] = pid;
      else
        pidR[npR++] = pid;
    }
    if(npL > Ngmin) group_bt_kokkos(pidL,npL,icell); // need to pass particle list above
    if(npR > Ngmin) group_bt_kokkos(pidR,npR,icell);
#endif
  }
}
