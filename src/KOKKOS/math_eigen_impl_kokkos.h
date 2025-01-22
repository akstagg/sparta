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

/* ----------------------------------------------------------------------
   Contributing authors: Andrew Jewett (Scripps Research, Jacobi algorithm)
------------------------------------------------------------------------- */

#ifndef SPARTA_MATH_EIGEN_IMPL_KOKKOS_H
#define SPARTA_MATH_EIGEN_IMPL_KOKKOS_H

//        This file contains a library of functions and classes which can
//        efficiently perform eigendecomposition for an extremely broad
//        range of matrix types: both real and complex, dense and sparse.
//        Matrices need not be of type "double **", for example.
//        In principle, almost any type of C++ container can be used.
//        Some general C++11 compatible functions for allocating matrices and
//        calculating norms of real and complex vectors are also provided.
// note
//        The "Jacobi" and "PEigenDense" classes are used for calculating
//        eigenvalues and eigenvectors of conventional dense square matrices.

//#include <cassert>
#include <numeric>
#include <complex>
#include <limits>
#include <cmath>
#include <vector>
#include <random>
#include <functional>

namespace MathEigenKokkos {


  // ---- Eigendecomposition of small dense symmetric matrices ----

  /// @class Jacobi
  /// @brief Calculate the eigenvalues and eigenvectors of a symmetric matrix
  ///        using the Jacobi eigenvalue algorithm.  This code (along with tests
  ///        and benchmarks) is available free of license restrictions at:
  ///        https://github.com/jewettaij/jacobi_pd
  /// @note  The "Vector", "Matrix", and "ConstMatrix" type arguments can be any
  ///     C or C++ object that supports indexing, including pointers or vectors.

  template<typename Scalar,
           typename Vector,
           typename Matrix,
           typename ConstMatrix=Matrix>

  class Jacobi_kk
  {
    int n;            //!< the size of the matrices you want to diagonalize
    Scalar **M;       //!< local copy of the current matrix being analyzed
    // Precomputed cosine, sine, and tangent of the most recent rotation angle:
    Scalar c;         //!< = cos(θ)
    Scalar s;         //!< = sin(θ)
    Scalar t;         //!< = tan(θ),  (note |t|<=1)
    int *max_idx_row; //!< = keep track of the the maximum element in row i (>i)

  public:

    // @typedef choose the criteria for sorting eigenvalues and eigenvectors
    typedef enum eSortCriteria {
      DO_NOT_SORT,
      SORT_DECREASING_EVALS,
      SORT_INCREASING_EVALS,
      SORT_DECREASING_ABS_EVALS,
      SORT_INCREASING_ABS_EVALS
    } SortCriteria;

    /// @brief Calculate the eigenvalues and eigenvectors of a symmetric matrix
    ///        using the Jacobi eigenvalue algorithm.
    /// @returns 0 if the algorithm converged,
    ///          1 if the algorithm failed to converge. (IE, if the number of
    ///            pivot iterations exceeded max_num_sweeps * iters_per_sweep,
    ///            where iters_per_sweep = (n*(n-1)/2))
    /// @note  To reduce the computation time further, set calc_evecs=false.

    KOKKOS_INLINE_FUNCTION
    int Diagonalize(ConstMatrix mat, //!< the matrix you wish to diagonalize (size n)
                    Vector eval,     //!< store the eigenvalues here
                    Matrix evec,     //!< store the eigenvectors here (in rows)
                    SortCriteria sort_criteria=SORT_DECREASING_EVALS,//!<sort results?
                    bool calc_evecs=true,    //!< calculate the eigenvectors?
                    int max_num_sweeps=50    //!< limit the number of iterations
                    );

    //        Space is allocated for the "M" and "max_idx_row" arrays on the stack
    //        in advance and passed to the constructor.
    //        n  the size (ie. number of rows) of the (square) matrix.
    //        M            optional preallocated n x n array
    //        max_idx_row  optional preallocated array of size n
    //  note  If either the "M" or "max_idx_row" arguments are specified,
    //        they both must be specified.
    KOKKOS_INLINE_FUNCTION
    Jacobi_kk(int n, Scalar **M, int *max_idx_row);

  private:
    bool is_preallocated;

    // (Descriptions of private functions can be found in their implementation.)
    KOKKOS_INLINE_FUNCTION
    void CalcRot(Scalar const *const *M, int i, int j);

    KOKKOS_INLINE_FUNCTION
    void ApplyRot(Scalar **M, int i, int j);

    KOKKOS_INLINE_FUNCTION
    void ApplyRotLeft(Matrix E, int i, int j);

    KOKKOS_INLINE_FUNCTION
    int MaxEntryRow(Scalar const *const *M, int i) const;

    KOKKOS_INLINE_FUNCTION
    void MaxEntry(Scalar const *const *M, int& i_max, int& j_max) const;

    KOKKOS_INLINE_FUNCTION
    void SortRows(Vector v, Matrix M, int n, SortCriteria s=SORT_DECREASING_EVALS) const;

  }; // class Jacobi_kk

// --------------------------------------
// ----------- IMPLEMENTATION -----------
// --------------------------------------


// --- Implementation: Eigendecomposition of small dense matrices ---

template<typename Scalar,typename Vector,typename Matrix,typename ConstMatrix>
KOKKOS_INLINE_FUNCTION
Jacobi_kk<Scalar, Vector, Matrix, ConstMatrix>::
Jacobi_kk(int n, Scalar **M, int *max_idx_row) {
  is_preallocated = true;
  this->n = n;
  this->M = M;
  this->max_idx_row = max_idx_row;
}

template<typename Scalar,typename Vector,typename Matrix,typename ConstMatrix>
KOKKOS_INLINE_FUNCTION
int Jacobi_kk<Scalar, Vector, Matrix, ConstMatrix>::
Diagonalize(ConstMatrix mat,    // the matrix you wish to diagonalize (size n)
            Vector eval,        // store the eigenvalues here
            Matrix evec,        // store the eigenvectors here (in rows)
            SortCriteria sort_criteria, // sort results?
            bool calc_evec,     // calculate the eigenvectors?
            int max_num_sweeps) // limit the number of iterations ("sweeps")
{
  // -- Initialization --
  for (int i = 0; i < n; i++)
    for (int j = i; j < n; j++)          //copy mat[][] into M[][]
      M[i][j] = mat[i][j];               //(M[][] is a local copy we can modify)

  if (calc_evec)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        evec[i][j] = (i==j) ? 1.0 : 0.0; //Set evec equal to the identity matrix

  for (int i = 0; i < n-1; i++)          //Initialize the "max_idx_row[]" array
    max_idx_row[i] = MaxEntryRow(M, i);  //(which is needed by MaxEntry())

  // -- Iteration --
  int n_iters;
  int max_num_iters = max_num_sweeps*n*(n-1)/2; //"sweep" = n*(n-1)/2 iters
  for (n_iters=0; n_iters < max_num_iters; n_iters++) {
    int i,j;
    MaxEntry(M, i, j); // Find the maximum entry in the matrix. Store in i,j

    // If M[i][j] is small compared to M[i][i] and M[j][j], set it to 0.
    if ((M[i][i] + M[i][j] == M[i][i]) && (M[j][j] + M[i][j] == M[j][j])) {
      M[i][j] = 0.0;
      max_idx_row[i] = MaxEntryRow(M,i); //must also update max_idx_row[i]
    }

    if (M[i][j] == 0.0)
      break;

    // Otherwise, apply a rotation to make M[i][j] = 0
    CalcRot(M, i, j);  // Calculate the parameters of the rotation matrix.
    ApplyRot(M, i, j); // Apply this rotation to the M matrix.
    if (calc_evec)     // Optional: If the caller wants the eigenvectors, then
      ApplyRotLeft(evec,i,j); // apply the rotation to the eigenvector matrix

  } //for (int n_iters=0; n_iters < max_num_iters; n_iters++)

  // -- Post-processing --
  for (int i = 0; i < n; i++)
    eval[i] = M[i][i];

  // Optional: Sort results by eigenvalue.
  SortRows(eval, evec, n, sort_criteria);

  return (n_iters == max_num_iters);
}


/// brief  Calculate the components of a rotation matrix which performs a
///        rotation in the i,j plane by an angle (θ) that (when multiplied on
///        both sides) will zero the ij'th element of M, so that afterwards
///        M[i][j] = 0.  The results will be stored in c, s, and t
///        (which store cos(θ), sin(θ), and tan(θ), respectively).

template<typename Scalar,typename Vector,typename Matrix,typename ConstMatrix>
KOKKOS_INLINE_FUNCTION
void Jacobi_kk<Scalar, Vector, Matrix, ConstMatrix>::
CalcRot(Scalar const *const *M,    //!< matrix
        int i,       //!< row index
        int j)       //!< column index
{
  t = 1.0; // = tan(θ)
  Scalar M_jj_ii = (M[j][j] - M[i][i]);
  if (M_jj_ii != 0.0) {
    // kappa = (M[j][j] - M[i][i]) / (2*M[i][j])
    Scalar kappa = M_jj_ii;
    t = 0.0;
    Scalar M_ij = M[i][j];
    if (M_ij != 0.0) {
      kappa /= (2.0*M_ij);
      // t satisfies: t^2 + 2*t*kappa - 1 = 0
      // (choose the root which has the smaller absolute value)
      t = 1.0 / (sqrt(1 + kappa*kappa) + fabs(kappa));
      if (kappa < 0.0)
        t = -t;
    }
  }
  //assert(std::abs(t) <= 1.0);
  c = 1.0 / sqrt(1 + t*t);
  s = c*t;
}


/// brief   Perform a similarity transformation by multiplying matrix M on both
///         sides by a rotation matrix (and its transpose) to eliminate M[i][j].
/// details This rotation matrix performs a rotation in the i,j plane by
///         angle θ.  This function assumes that c=cos(θ). s=som(θ), t=tan(θ)
///         have been calculated previously (using the CalcRot() function).
///         It also assumes that i<j.  The max_idx_row[] array is also updated.
///         To save time, since the matrix is symmetric, the elements
///         below the diagonal (ie. M[u][v] where u>v) are not computed.
///
/// verbatim
///
///   M' = R^T * M * R
/// where R the rotation in the i,j plane and ^T denotes the transpose.
///                 i         j
///       _                             _
///      |  1                            |
///      |    .                          |
///      |      .                        |
///      |        1                      |
///      |          c   ...   s          |
///      |          .  .      .          |
/// R  = |          .    1    .          |
///      |          .      .  .          |
///      |          -s  ...   c          |
///      |                      1        |
///      |                        .      |
///      |                          .    |
///      |_                           1 _|
///
/// endverbatim
///
/// Let M' denote the matrix M after multiplication by R^T and R.
/// The components of M' are:
///   M'_uv =  Σ_w  Σ_z   R_wu * M_wz * R_zv
///
/// note
/// The rotation at location i,j will modify all of the matrix
/// elements containing at least one index which is either i or j
/// such as: M[w][i], M[i][w], M[w][j], M[j][w].
/// Check and see whether these modified matrix elements exceed the
/// corresponding values in max_idx_row[] array for that row.
/// If so, then update max_idx_row for that row.
/// This is somewhat complicated by the fact that we must only consider
/// matrix elements in the upper-right triangle strictly above the diagonal.
/// (ie. matrix elements whose second index is > the first index).

template<typename Scalar,typename Vector,typename Matrix,typename ConstMatrix>
KOKKOS_INLINE_FUNCTION
void Jacobi_kk<Scalar, Vector, Matrix, ConstMatrix>::
ApplyRot(Scalar **M,  // matrix
         int i,       // row index
         int j)       // column index
{
  // Recall that c = cos(θ), s = sin(θ), t = tan(θ) (and t <= 1.0)

  // Compute the diagonal elements of M which have changed:
  M[i][i] -= t * M[i][j];
  M[j][j] += t * M[i][j];

  //Update the off-diagonal elements of M which will change (above the diagonal)
  //assert(i < j);
  M[i][j] = 0.0;

  //compute M[w][i] and M[i][w] for all w!=i,considering above-diagonal elements
  for (int w=0; w < i; w++) {        // 0 <= w <  i  <  j < n
    M[i][w] = M[w][i]; //backup the previous value. store below diagonal (i>w)
    M[w][i] = c*M[w][i] - s*M[w][j]; //M[w][i], M[w][j] from previous iteration
    if (i == max_idx_row[w]) max_idx_row[w] = MaxEntryRow(M, w);
    else if (fabs(M[w][i])>fabs(M[w][max_idx_row[w]])) max_idx_row[w]=i;
    //assert(max_idx_row[w] == MaxEntryRow(M, w));
  }
  for (int w=i+1; w < j; w++) {      // 0 <= i <  w  <  j < n
    M[w][i] = M[i][w]; //backup the previous value. store below diagonal (w>i)
    M[i][w] = c*M[i][w] - s*M[w][j]; //M[i][w], M[w][j] from previous iteration
  }
  for (int w=j+1; w < n; w++) {      // 0 <= i < j+1 <= w < n
    M[w][i] = M[i][w]; //backup the previous value. store below diagonal (w>i)
    M[i][w] = c*M[i][w] - s*M[j][w]; //M[i][w], M[j][w] from previous iteration
  }

  // now that we're done modifying row i, we can update max_idx_row[i]
  max_idx_row[i] = MaxEntryRow(M, i);

  //compute M[w][j] and M[j][w] for all w!=j,considering above-diagonal elements
  for (int w=0; w < i; w++) {        // 0 <=  w  <  i <  j < n
    M[w][j] = s*M[i][w] + c*M[w][j]; //M[i][w], M[w][j] from previous iteration
    if (j == max_idx_row[w]) max_idx_row[w] = MaxEntryRow(M, w);
    else if (fabs(M[w][j])>fabs(M[w][max_idx_row[w]])) max_idx_row[w]=j;
    //assert(max_idx_row[w] == MaxEntryRow(M, w));
  }
  for (int w=i+1; w < j; w++) {      // 0 <= i+1 <= w <  j < n
    M[w][j] = s*M[w][i] + c*M[w][j]; //M[w][i], M[w][j] from previous iteration
    if (j == max_idx_row[w]) max_idx_row[w] = MaxEntryRow(M, w);
    else if (fabs(M[w][j])>fabs(M[w][max_idx_row[w]])) max_idx_row[w]=j;
    //assert(max_idx_row[w] == MaxEntryRow(M, w));
  }
  for (int w=j+1; w < n; w++) {      // 0 <=  i  <  j <  w < n
    M[j][w] = s*M[w][i] + c*M[j][w]; //M[w][i], M[j][w] from previous iteration
  }
  // now that we're done modifying row j, we can update max_idx_row[j]
  max_idx_row[j] = MaxEntryRow(M, j);
} //Jacobi::ApplyRot()


/// brief  Multiply matrix E on the left by the (previously calculated)
///        rotation matrix.
///
/// details
/// Multiply matrix M on the LEFT side by a transposed rotation matrix, R^T.
/// This matrix performs a rotation in the i,j plane by angle θ
/// (where the arguments "s" and "c" refer to cos(θ) and sin(θ), respectively).
///
/// verbatim
///   E'_uv = Σ_w  R_wu * E_wv
/// endverbatim

template<typename Scalar,typename Vector,typename Matrix,typename ConstMatrix>
KOKKOS_INLINE_FUNCTION
void Jacobi_kk<Scalar, Vector, Matrix, ConstMatrix>::
ApplyRotLeft(Matrix E,  // matrix
             int i,     // row index
             int j)     // column index
{
  // recall that c = cos(θ) and s = sin(θ)
  for (int v = 0; v < n; v++) {
    Scalar Eiv = E[i][v]; //backup E[i][v]
    E[i][v] = c*E[i][v] - s*E[j][v];
    E[j][v] = s*Eiv     + c*E[j][v];
  }
}

/// brief  Find the off-diagonal index in row i whose absolute value is largest
template<typename Scalar,typename Vector,typename Matrix,typename ConstMatrix>
KOKKOS_INLINE_FUNCTION
int Jacobi_kk<Scalar, Vector, Matrix, ConstMatrix>::
MaxEntryRow(Scalar const *const *M, int i) const {
  int j_max = i+1;
  for (int j = i+2; j < n; j++)
    if (fabs(M[i][j]) > fabs(M[i][j_max]))
      j_max = j;
  return j_max;
}

/// brief  Find the indices (i_max, j_max) marking the location of the
///        entry in the matrix with the largest absolute value.  This
///        uses the max_idx_row[] array to find the answer in O(n) time.
/// returns  This function does not return a avalue.  However after it is
///          invoked, the location of the largest matrix element will be
///          stored in the i_max and j_max arguments.
template<typename Scalar,typename Vector,typename Matrix,typename ConstMatrix>
KOKKOS_INLINE_FUNCTION
void Jacobi_kk<Scalar, Vector, Matrix, ConstMatrix>::
MaxEntry(Scalar const *const *M, int& i_max, int& j_max) const {
  // find the maximum entry in the matrix M in O(n) time
  i_max = 0;
  j_max = max_idx_row[i_max];
  Scalar max_entry = fabs(M[i_max][j_max]);
  int nm1 = n-1;
  for (int i=1; i < nm1; i++) {
    int j = max_idx_row[i];
    if (fabs(M[i][j]) > max_entry) {
      max_entry = fabs(M[i][j]);
      i_max = i;
      j_max = j;
    }
  }
}

/// brief  Sort the rows in matrix "evec" according to the numbers in "eval".
template<typename Scalar,typename Vector,typename Matrix,typename ConstMatrix>
KOKKOS_INLINE_FUNCTION
void Jacobi_kk<Scalar, Vector, Matrix, ConstMatrix>::
SortRows(Vector eval,        // vector containing the keys used for sorting
         Matrix evec,        // matrix whose rows will be sorted according to v
         int n,              // size of the vector and matrix
         SortCriteria sort_criteria) const // sort eigenvalues?
{
  for (int i = 0; i < n-1; i++) {
    int i_max = i;
    for (int j = i+1; j < n; j++) {
      // find the "maximum" element in the array starting at position i+1
      switch (sort_criteria) {
      case SORT_DECREASING_EVALS:
        if (eval[j] > eval[i_max])
          i_max = j;
        break;
      case SORT_INCREASING_EVALS:
        if (eval[j] < eval[i_max])
          i_max = j;
        break;
      case SORT_DECREASING_ABS_EVALS:
        if (fabs(eval[j]) > fabs(eval[i_max]))
          i_max = j;
        break;
      case SORT_INCREASING_ABS_EVALS:
        if (fabs(eval[j]) < fabs(eval[i_max]))
          i_max = j;
        break;
      default:
        break;
      }
    }

    // sort "eval"
    Scalar val;
    val = eval[i_max];
    eval[i_max] = eval[i];
    eval[i] = val;

    // sort "evec"
    for (int k = 0; k < n; k++) {
      val = evec[i_max][k];
      evec[i_max][k] = evec[i][k];
      evec[i][k] = val;
    }
  }
}

} //namespace MathEigenKokkos

#endif //#ifndef SPARTA_MATH_EIGEN_IMPL_KOKKOS_H
