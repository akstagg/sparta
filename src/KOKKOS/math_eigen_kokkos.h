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

#ifndef SPARTA_MATH_EIGEN_KOKKOS_H
#define SPARTA_MATH_EIGEN_KOKKOS_H

#include "kokkos_base.h"
#include "kokkos_type.h"
#include "math_eigen_kokkos.h"
#include "math_eigen_impl_kokkos.h"

namespace MathEigenKokkos {

/** A specialized function which finds the eigenvalues and eigenvectors
 *  of a 3x3 matrix.
 *
 * \param  mat   the 3x3 matrix you wish to diagonalize
 * \param  eval  store the eigenvalues here
 * \param  evec  store the eigenvectors here...
 * \return       0 if eigenvalue calculation converged, 1 if it failed */

typedef Jacobi_kk<double, double *, double (*)[3], double const (*)[3]> Jacobi_v1_kk;

KOKKOS_INLINE_FUNCTION
int jacobi3_kokkos(double const mat[3][3], double *eval, double evec[3][3]) {
  // make copy of const matrix

  double mat_cpy[3][3] = {{mat[0][0], mat[0][1], mat[0][2]},
                          {mat[1][0], mat[1][1], mat[1][2]},
                          {mat[2][0], mat[2][1], mat[2][2]}};

  double *M[3] = {&(mat_cpy[0][0]), &(mat_cpy[1][0]), &(mat_cpy[2][0])};
  int midx[3];

  // create instance of generic Jacobi class and get eigenvalues and -vectors

  Jacobi_v1_kk ecalc3_kk(3, M, midx);
  int ierror = ecalc3_kk.Diagonalize(mat, eval, evec, Jacobi_v1_kk::SORT_DECREASING_EVALS);

  // transpose the evec matrix

  double val;
  for (int i = 0; i < 3; i++) {
    for (int j = i + 1; j < 3; j++) {
      val = evec[j][i];
      evec[j][i] = evec[i][j];
      evec[i][j] = val;
    }
  }
  return ierror;
}

}    // namespace MathEigenKokkos

#endif    //#ifndef SPARTA_MATH_EIGEN_KOKKOS_H
