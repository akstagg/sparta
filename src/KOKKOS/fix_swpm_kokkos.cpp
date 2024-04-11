/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "fix_swpm_kokkos.h"
#include "update.h"
#include "particle.h"
#include "memory_kokkos.h"
#include "error.h"
#include "sparta_masks.h"


using namespace SPARTA_NS;

enum{INT,DOUBLE};                      // several files

/* ---------------------------------------------------------------------- */

FixSWPMKokkos::FixSWPMKokkos(SPARTA *sparta, int narg, char **arg) :
  FixSWPM(sparta, narg, arg)
{
  kokkos_flag = 1;
}

/* ---------------------------------------------------------------------- */

FixSWPMKokkos::~FixSWPMKokkos()
{
  if (copymode) return;
}

/* ---------------------------------------------------------------------- */

void FixSWPMKokkos::pre_update_custom_kokkos()
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
  d_particles = particle_kk->k_particles.d_view;
  d_species = particle_kk->k_species.d_view;
  auto h_ewhich = particle_kk->k_ewhich.h_view;
  auto k_edvec = particle_kk->k_edvec;
  d_sweights = k_edvec.h_view[h_ewhich[index_sweight]].k_view.d_view;
}

/* ----------------------------------------------------------------------
   called when a particle with index is created
------------------------------------------------------------------------- */

void FixSWPMKokkos::update_custom(int index, double temp_thermal,
                                  double temp_rot, double temp_vib,
                                  double *vstream)
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Host,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
  FixSWPM::update_custom(index, temp_thermal, temp_rot, temp_vib, vstream);
  particle_kk->modify(Host,CUSTOM_MASK);
}
