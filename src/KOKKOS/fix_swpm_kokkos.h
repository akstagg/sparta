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

#ifdef FIX_CLASS

FixStyle(swpm/kk,FixSWPMKokkos)

#else

#ifndef SPARTA_FIX_SWPM_KOKKOS_H
#define SPARTA_FIX_SWPM_KOKKOS_H

#include "fix_swpm.h"
#include "particle_kokkos.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

  class FixSWPMKokkos : public FixSWPM {
  public:

    FixSWPMKokkos(class SPARTA *, int, char **);
    ~FixSWPMKokkos();
    void pre_update_custom_kokkos();
    void update_custom(int, double, double, double, double *);


  private:

    t_particle_1d d_particles; // may not need this
    t_species_1d d_species; // may not need this
    DAT::t_float_1d d_sweights;

};

}

#endif
#endif

/* ERROR/WARNING messages:
 */

