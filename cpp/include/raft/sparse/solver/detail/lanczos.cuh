/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "raft/core/device_mdspan.hpp"
#include "raft/core/host_mdspan.hpp"
#include "raft/core/mdspan_types.hpp"
#include <raft/core/device_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/normalize.cuh>
#include <raft/linalg/norm_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/random/rng.cuh>

// for cmath:
#include <raft/core/logger-macros.hpp>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#define _USE_MATH_DEFINES

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/spectral/detail/lapack.hpp>
#include <raft/spectral/detail/warn_dbg.hpp>
#include <raft/spectral/matrix_wrappers.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda.h>

#include <curand.h>

#include <cmath>
#include <vector>

namespace raft::sparse::solver::detail {

// curandGeneratorNormalX
inline curandStatus_t curandGenerateNormalX(
  curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev)
{
  return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}
inline curandStatus_t curandGenerateNormalX(
  curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev)
{
  return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}

// =========================================================
// Helper functions
// =========================================================

/**
 *  @brief  Perform Lanczos iteration
 *    Lanczos iteration is performed on a shifted matrix A+shift*I.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param A Matrix.
 *  @param iter Pointer to current Lanczos iteration. On exit, the
 *    variable is set equal to the final Lanczos iteration.
 *  @param maxIter Maximum Lanczos iteration. This function will
 *    perform a maximum of maxIter-*iter iterations.
 *  @param shift Matrix shift.
 *  @param tol Convergence tolerance. Lanczos iteration will
 *    terminate when the residual norm (i.e. entry in beta_host) is
 *    less than tol.
 *  @param reorthogonalize Whether to reorthogonalize Lanczos
 *    vectors.
 *  @param alpha_host (Output, host memory, maxIter entries)
 *    Diagonal entries of Lanczos system.
 *  @param beta_host (Output, host memory, maxIter entries)
 *    Off-diagonal entries of Lanczos system.
 *  @param lanczosVecs_dev (Input/output, device memory,
 *    n*(maxIter+1) entries) Lanczos vectors. Vectors are stored as
 *    columns of a column-major matrix with dimensions
 *    n x (maxIter+1).
 *  @param work_dev (Output, device memory, maxIter entries)
 *    Workspace. Not needed if full reorthogonalization is disabled.
 *  @return Zero if successful. Otherwise non-zero.
 */
template <typename index_type_t, typename value_type_t>
int performLanczosIteration(raft::resources const& handle,
                            spectral::matrix::sparse_matrix_t<index_type_t, value_type_t> const* A,
                            index_type_t* iter,
                            index_type_t maxIter,
                            value_type_t shift,
                            value_type_t tol,
                            bool reorthogonalize,
                            value_type_t* __restrict__ alpha_host,
                            value_type_t* __restrict__ beta_host,
                            value_type_t* __restrict__ lanczosVecs_dev,
                            value_type_t* __restrict__ work_dev,
                            value_type_t* __restrict__ work_host,
                            value_type_t* __restrict__ Z_host,
                            value_type_t* __restrict__ eigVals_dev,
                            index_type_t nEigVecs,
                            index_type_t* totalIter,
                            int conv_n_iters,
                            float conv_eps)
{
  // -------------------------------------------------------
  // Variable declaration
  // -------------------------------------------------------

  // Useful variables
  constexpr value_type_t one    = 1;
  constexpr value_type_t negOne = -1;
  constexpr value_type_t zero   = 0;
  value_type_t alpha;

  auto cublas_h = resource::get_cublas_handle(handle);
  auto stream   = resource::get_cuda_stream(handle);

  RAFT_EXPECTS(A != nullptr, "Null matrix pointer.");

  index_type_t n = A->nrows_;

  // -------------------------------------------------------
  // Compute second Lanczos vector
  // -------------------------------------------------------
  if (*iter <= 0) {
    *iter = 1;

    // Apply matrix
    if (shift != 0)
      RAFT_CUDA_TRY(cudaMemcpyAsync(lanczosVecs_dev + n,
                                    lanczosVecs_dev,
                                    n * sizeof(value_type_t),
                                    cudaMemcpyDeviceToDevice,
                                    stream));
    A->mv(1, lanczosVecs_dev, shift, lanczosVecs_dev + n);

    // Orthogonalize Lanczos vector
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(
      cublas_h, n, lanczosVecs_dev, 1, lanczosVecs_dev + IDX(0, 1, n), 1, alpha_host, stream));

    alpha = -alpha_host[0];
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasaxpy(
      cublas_h, n, &alpha, lanczosVecs_dev, 1, lanczosVecs_dev + IDX(0, 1, n), 1, stream));
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasnrm2(
      cublas_h, n, lanczosVecs_dev + IDX(0, 1, n), 1, beta_host, stream));

    // Check if Lanczos has converged
    if (beta_host[0] <= tol) return 0;

    // Normalize Lanczos vector
    alpha = 1 / beta_host[0];
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasscal(
      cublas_h, n, &alpha, lanczosVecs_dev + IDX(0, 1, n), 1, stream));
  }

  // -------------------------------------------------------
  // Compute remaining Lanczos vectors
  // -------------------------------------------------------
  // int orig_conv_n_iters = conv_n_iters;
  // std::vector<value_type_t> prev_conv(nEigVecs, 0);
  int status = 0;


  while (*iter < maxIter && conv_n_iters > 0) {
    ++(*iter);

    // Apply matrix
    if (shift != 0)
      RAFT_CUDA_TRY(cudaMemcpyAsync(lanczosVecs_dev + (*iter) * n,
                                    lanczosVecs_dev + (*iter - 1) * n,
                                    n * sizeof(value_type_t),
                                    cudaMemcpyDeviceToDevice,
                                    stream));
    A->mv(1, lanczosVecs_dev + IDX(0, *iter - 1, n), shift, lanczosVecs_dev + IDX(0, *iter, n));

    // Full reorthogonalization
    //   "Twice is enough" algorithm per Kahan and Parlett
    if (!reorthogonalize) {
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(cublas_h,
                                                       CUBLAS_OP_T,
                                                       n,
                                                       *iter,
                                                       &one,
                                                       lanczosVecs_dev,
                                                       n,
                                                       lanczosVecs_dev + IDX(0, *iter, n),
                                                       1,
                                                       &zero,
                                                       work_dev,
                                                       1,
                                                       stream));

      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(cublas_h,
                                                       CUBLAS_OP_N,
                                                       n,
                                                       *iter,
                                                       &negOne,
                                                       lanczosVecs_dev,
                                                       n,
                                                       work_dev,
                                                       1,
                                                       &one,
                                                       lanczosVecs_dev + IDX(0, *iter, n),
                                                       1,
                                                       stream));

      RAFT_CUDA_TRY(cudaMemcpyAsync(alpha_host + (*iter - 1),
                                    work_dev + (*iter - 1),
                                    sizeof(value_type_t),
                                    cudaMemcpyDeviceToHost,
                                    stream));

      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(cublas_h,
                                                       CUBLAS_OP_T,
                                                       n,
                                                       *iter,
                                                       &one,
                                                       lanczosVecs_dev,
                                                       n,
                                                       lanczosVecs_dev + IDX(0, *iter, n),
                                                       1,
                                                       &zero,
                                                       work_dev,
                                                       1,
                                                       stream));

      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(cublas_h,
                                                       CUBLAS_OP_N,
                                                       n,
                                                       *iter,
                                                       &negOne,
                                                       lanczosVecs_dev,
                                                       n,
                                                       work_dev,
                                                       1,
                                                       &one,
                                                       lanczosVecs_dev + IDX(0, *iter, n),
                                                       1,
                                                       stream));
    }

    // Orthogonalization with 3-term recurrence relation
    else {
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(cublas_h,
                                                      n,
                                                      lanczosVecs_dev + IDX(0, *iter - 1, n),
                                                      1,
                                                      lanczosVecs_dev + IDX(0, *iter, n),
                                                      1,
                                                      alpha_host + (*iter - 1),
                                                      stream));

      // print_host_vector("cublasdot alpha_host", alpha_host, 306, std::cout);

      auto alpha = -alpha_host[*iter - 1];
      // print_device_vector("lanczosVecs_dev", lanczosVecs_dev, 5, std::cout);
      // std::cout << "alpha " << alpha << std::endl;
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasaxpy(cublas_h,
                                                       n,
                                                       &alpha,
                                                       lanczosVecs_dev + IDX(0, *iter - 1, n),
                                                       1,
                                                       lanczosVecs_dev + IDX(0, *iter, n),
                                                       1,
                                                       stream));

      alpha = -beta_host[*iter - 2];
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasaxpy(cublas_h,
                                                       n,
                                                       &alpha,
                                                       lanczosVecs_dev + IDX(0, *iter - 2, n),
                                                       1,
                                                       lanczosVecs_dev + IDX(0, *iter, n),
                                                       1,
                                                       stream));
    }

    // Compute residual
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasnrm2(
      cublas_h, n, lanczosVecs_dev + IDX(0, *iter, n), 1, beta_host + *iter - 1, stream));

    // Check if Lanczos has converged
    if (beta_host[*iter - 1] <= tol) break;
    // std::cout << (*iter - 1) << " " << beta_host[*iter - 1] << " " << tol << " " << std::endl;

    // Normalize Lanczos vector
    alpha = 1 / beta_host[*iter - 1];
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasscal(
      cublas_h, n, &alpha, lanczosVecs_dev + IDX(0, *iter, n), 1, stream));

    // index_type_t* effIter = iter;
    // // Solve tridiagonal system
    // memcpy(work_host + 2 * (*effIter), alpha_host, (*effIter) * sizeof(value_type_t));
    // memcpy(work_host + 3 * (*effIter), beta_host, (*effIter - 1) * sizeof(value_type_t));
    // Lapack<value_type_t>::steqr('I',
    //                             *effIter,
    //                             work_host + 2 * (*effIter),
    //                             work_host + 3 * (*effIter),
    //                             Z_host,
    //                             *effIter,
    //                             work_host);

    // // Obtain desired eigenvalues by applying shift
    // for (int i = 0; i < *effIter; ++i)
    //   work_host[i + 2 * (*effIter)] -= shift;
    // for (int i = *effIter; i < nEigVecs; ++i)
    //   work_host[i + 2 * (*effIter)] = 0;

    // // Copy results to device memory
    // RAFT_CUDA_TRY(cudaMemcpyAsync(eigVals_dev,
    //                               work_host + 2 * (*effIter),
    //                               nEigVecs * sizeof(value_type_t),
    //                               cudaMemcpyHostToDevice,
    //                               stream));

    // RAFT_CUDA_TRY(cudaMemcpyAsync(work_dev,
    //                             Z_host,
    //                             (*effIter) * nEigVecs * sizeof(value_type_t),
    //                             cudaMemcpyHostToDevice,
    //                             stream));

    // std::ofstream out_file("output2.txt"); // Open a file for writing
  
    // // Check if the file is open
    // if (!out_file.is_open()) {
    //   std::cerr << "Failed to open output file!" << std::endl;
    // }

    // // std::cout << ((*effIter) * (nEigVecs - 1)) << std::endl;


    // // print_device_vector("work_dev", work_dev, (*effIter) * nEigVecs, out_file);
    // // raft::device_vector_view<value_type_t, int, raft::col_major> work_dev_view = raft::make_device_vector_view(work_dev, (*effIter) * nEigVecs);
    // raft::host_matrix_view<value_type_t, uint32_t, raft::col_major> Z_host_view = raft::make_host_matrix_view<value_type_t, uint32_t, raft::col_major>(Z_host, (*effIter), nEigVecs);
    // std::vector<value_type_t> Z_host_last_row(nEigVecs);
    // for (int i = 0; i < nEigVecs; i++) {
    //   Z_host_last_row[i] = beta_host[*iter - 1] * Z_host_view((*effIter) - 1, i);
    // }
    // float Znorm = 0;
    // for (int i = 0; i < nEigVecs; i++) {
    //   Znorm += Z_host_last_row[i] * Z_host_last_row[i];
    // }
    // std::sqrt(Znorm);
    // std::cout << (*totalIter + (*iter - 1)) << " " << Znorm << " " << tol << " " << beta_host[*iter - 1] << " " << std::endl;
    // print_host_vector("Z_host_last_row", Z_host_last_row.data(), nEigVecs, out_file);
    // print_host_vector("work_dev", Z_host, (*effIter) * nEigVecs, out_file);

    // if (Znorm <= tol) {
    //   status = 1;
    //   break;
    // }
    // print_device_vector("work_dev", work_dev + ((*effIter - 1) * (nEigVecs)), nEigVecs, out_file);

    // print_device_vector("lanczos iteration", eigVals_dev, nEigVecs, out_file);
    // std::cout << *totalIter << " " << *effIter << std::endl;
    // std::cout << *totalIter << " " << *effIter << " " << iter_new << " " << maxIter_curr << " "<< (beta_host[*effIter - 1] > tol * fabs(shiftLower)) << std::endl;

    // value_type_t curr_conv;
    // std::vector<value_type_t> curr_conv(nEigVecs, 0);
    // raft::copy(curr_conv.data(), eigVals_dev, nEigVecs, stream);

    // for (int i = 0; i < nEigVecs; i++) {
    //   if (fabs(curr_conv[i] - prev_conv[i]) > conv_eps) {
    //     conv_n_iters = orig_conv_n_iters;
    //     raft::copy(prev_conv.data(), eigVals_dev, nEigVecs, stream);
    //     break;
    //   }
    //   if (i == nEigVecs - 1) {
    //     conv_n_iters -= 1;
    //     if (conv_n_iters == 0) status = 1;
    //   }
    // }
  }

  resource::sync_stream(handle, stream);

  return status;
}

/**
 *  @brief  Find Householder transform for 3-dimensional system
 *    Given an input vector v=[x,y,z]', this function finds a
 *    Householder transform P such that P*v is a multiple of
 *    e_1=[1,0,0]'. The input vector v is overwritten with the
 *    Householder vector such that P=I-2*v*v'.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param v (Input/output, host memory, 3 entries) Input
 *    3-dimensional vector. On exit, the vector is set to the
 *    Householder vector.
 *  @param Pv (Output, host memory, 1 entry) First entry of P*v
 *    (here v is the input vector). Either equal to ||v||_2 or
 *    -||v||_2.
 *  @param P (Output, host memory, 9 entries) Householder transform
 *    matrix. Matrix dimensions are 3 x 3.
 */
template <typename index_type_t, typename value_type_t>
static void findHouseholder3(value_type_t* v, value_type_t* Pv, value_type_t* P)
{
  // Compute norm of vector
  *Pv = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

  // Choose whether to reflect to e_1 or -e_1
  //   This choice avoids catastrophic cancellation
  if (v[0] >= 0) *Pv = -(*Pv);
  v[0] -= *Pv;

  // Normalize Householder vector
  value_type_t normHouseholder = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (normHouseholder != 0) {
    v[0] /= normHouseholder;
    v[1] /= normHouseholder;
    v[2] /= normHouseholder;
  } else {
    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
  }

  // Construct Householder matrix
  index_type_t i, j;
  for (j = 0; j < 3; ++j)
    for (i = 0; i < 3; ++i)
      P[IDX(i, j, 3)] = -2 * v[i] * v[j];
  for (i = 0; i < 3; ++i)
    P[IDX(i, i, 3)] += 1;
}

/**
 *  @brief  Apply 3-dimensional Householder transform to 4 x 4 matrix
 *    The Householder transform is pre-applied to the top three rows
 *  of the matrix and post-applied to the left three columns. The
 *  4 x 4 matrix is intended to contain the bulge that is produced
 *  in the Francis QR algorithm.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param v (Input, host memory, 3 entries) Householder vector.
 *  @param A (Input/output, host memory, 16 entries) 4 x 4 matrix.
 */
template <typename index_type_t, typename value_type_t>
static void applyHouseholder3(const value_type_t* v, value_type_t* A)
{
  // Loop indices
  index_type_t i, j;
  // Dot product between Householder vector and matrix row/column
  value_type_t vDotA;

  // Pre-apply Householder transform
  for (j = 0; j < 4; ++j) {
    vDotA = 0;
    for (i = 0; i < 3; ++i)
      vDotA += v[i] * A[IDX(i, j, 4)];
    for (i = 0; i < 3; ++i)
      A[IDX(i, j, 4)] -= 2 * v[i] * vDotA;
  }

  // Post-apply Householder transform
  for (i = 0; i < 4; ++i) {
    vDotA = 0;
    for (j = 0; j < 3; ++j)
      vDotA += A[IDX(i, j, 4)] * v[j];
    for (j = 0; j < 3; ++j)
      A[IDX(i, j, 4)] -= 2 * vDotA * v[j];
  }
}

/**
 *  @brief  Perform one step of Francis QR algorithm
 *    Equivalent to two steps of the classical QR algorithm on a
 *    tridiagonal matrix.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param n Matrix dimension.
 *  @param shift1 QR algorithm shift.
 *  @param shift2 QR algorithm shift.
 *  @param alpha (Input/output, host memory, n entries) Diagonal
 *    entries of tridiagonal matrix.
 *  @param beta (Input/output, host memory, n-1 entries)
 *    Off-diagonal entries of tridiagonal matrix.
 *  @param V (Input/output, host memory, n*n entries) Orthonormal
 *    transforms from previous steps of QR algorithm. Matrix
 *    dimensions are n x n. On exit, the orthonormal transform from
 *    this Francis QR step is post-applied to the matrix.
 *  @param work (Output, host memory, 3*n entries) Workspace.
 *  @return Zero if successful. Otherwise non-zero.
 */
template <typename index_type_t, typename value_type_t>
static int francisQRIteration(index_type_t n,
                              value_type_t shift1,
                              value_type_t shift2,
                              value_type_t* alpha,
                              value_type_t* beta,
                              value_type_t* V,
                              value_type_t* work)
{
  // -------------------------------------------------------
  // Variable declaration
  // -------------------------------------------------------

  // Temporary storage of 4x4 bulge and Householder vector
  value_type_t bulge[16];

  // Householder vector
  value_type_t householder[3];
  // Householder matrix
  value_type_t householderMatrix[3 * 3];

  // Shifts are roots of the polynomial p(x)=x^2+b*x+c
  value_type_t b = -shift1 - shift2;
  value_type_t c = shift1 * shift2;

  // Loop indices
  index_type_t i, j, pos;
  // Temporary variable
  value_type_t temp;

  // -------------------------------------------------------
  // Implementation
  // -------------------------------------------------------

  // Compute initial Householder transform
  householder[0] = alpha[0] * alpha[0] + beta[0] * beta[0] + b * alpha[0] + c;
  householder[1] = beta[0] * (alpha[0] + alpha[1] + b);
  householder[2] = beta[0] * beta[1];
  findHouseholder3<index_type_t, value_type_t>(householder, &temp, householderMatrix);

  // Apply initial Householder transform to create bulge
  memset(bulge, 0, 16 * sizeof(value_type_t));
  for (i = 0; i < 4; ++i)
    bulge[IDX(i, i, 4)] = alpha[i];
  for (i = 0; i < 3; ++i) {
    bulge[IDX(i + 1, i, 4)] = beta[i];
    bulge[IDX(i, i + 1, 4)] = beta[i];
  }
  applyHouseholder3<index_type_t, value_type_t>(householder, bulge);
  Lapack<value_type_t>::gemm(false, false, n, 3, 3, 1, V, n, householderMatrix, 3, 0, work, n);
  memcpy(V, work, 3 * n * sizeof(value_type_t));

  // Chase bulge to bottom-right of matrix with Householder transforms
  for (pos = 0; pos < n - 4; ++pos) {
    // Move to next position
    alpha[pos]     = bulge[IDX(0, 0, 4)];
    householder[0] = bulge[IDX(1, 0, 4)];
    householder[1] = bulge[IDX(2, 0, 4)];
    householder[2] = bulge[IDX(3, 0, 4)];
    for (j = 0; j < 3; ++j)
      for (i = 0; i < 3; ++i)
        bulge[IDX(i, j, 4)] = bulge[IDX(i + 1, j + 1, 4)];
    bulge[IDX(3, 0, 4)] = 0;
    bulge[IDX(3, 1, 4)] = 0;
    bulge[IDX(3, 2, 4)] = beta[pos + 3];
    bulge[IDX(0, 3, 4)] = 0;
    bulge[IDX(1, 3, 4)] = 0;
    bulge[IDX(2, 3, 4)] = beta[pos + 3];
    bulge[IDX(3, 3, 4)] = alpha[pos + 4];

    // Apply Householder transform
    findHouseholder3<index_type_t, value_type_t>(householder, beta + pos, householderMatrix);
    applyHouseholder3<index_type_t, value_type_t>(householder, bulge);
    Lapack<value_type_t>::gemm(
      false, false, n, 3, 3, 1, V + IDX(0, pos + 1, n), n, householderMatrix, 3, 0, work, n);
    memcpy(V + IDX(0, pos + 1, n), work, 3 * n * sizeof(value_type_t));
  }

  // Apply penultimate Householder transform
  //   Values in the last row and column are zero
  alpha[n - 4]   = bulge[IDX(0, 0, 4)];
  householder[0] = bulge[IDX(1, 0, 4)];
  householder[1] = bulge[IDX(2, 0, 4)];
  householder[2] = bulge[IDX(3, 0, 4)];
  for (j = 0; j < 3; ++j)
    for (i = 0; i < 3; ++i)
      bulge[IDX(i, j, 4)] = bulge[IDX(i + 1, j + 1, 4)];
  bulge[IDX(3, 0, 4)] = 0;
  bulge[IDX(3, 1, 4)] = 0;
  bulge[IDX(3, 2, 4)] = 0;
  bulge[IDX(0, 3, 4)] = 0;
  bulge[IDX(1, 3, 4)] = 0;
  bulge[IDX(2, 3, 4)] = 0;
  bulge[IDX(3, 3, 4)] = 0;
  findHouseholder3<index_type_t, value_type_t>(householder, beta + n - 4, householderMatrix);
  applyHouseholder3<index_type_t, value_type_t>(householder, bulge);
  Lapack<value_type_t>::gemm(
    false, false, n, 3, 3, 1, V + IDX(0, n - 3, n), n, householderMatrix, 3, 0, work, n);
  memcpy(V + IDX(0, n - 3, n), work, 3 * n * sizeof(value_type_t));

  // Apply final Householder transform
  //   Values in the last two rows and columns are zero
  alpha[n - 3]   = bulge[IDX(0, 0, 4)];
  householder[0] = bulge[IDX(1, 0, 4)];
  householder[1] = bulge[IDX(2, 0, 4)];
  householder[2] = 0;
  for (j = 0; j < 3; ++j)
    for (i = 0; i < 3; ++i)
      bulge[IDX(i, j, 4)] = bulge[IDX(i + 1, j + 1, 4)];
  findHouseholder3<index_type_t, value_type_t>(householder, beta + n - 3, householderMatrix);
  applyHouseholder3<index_type_t, value_type_t>(householder, bulge);
  Lapack<value_type_t>::gemm(
    false, false, n, 2, 2, 1, V + IDX(0, n - 2, n), n, householderMatrix, 3, 0, work, n);
  memcpy(V + IDX(0, n - 2, n), work, 2 * n * sizeof(value_type_t));

  // Bulge has been eliminated
  alpha[n - 2] = bulge[IDX(0, 0, 4)];
  alpha[n - 1] = bulge[IDX(1, 1, 4)];
  beta[n - 2]  = bulge[IDX(1, 0, 4)];

  return 0;
}

/**
 *  @brief  Perform implicit restart of Lanczos algorithm
 *    Shifts are Chebyshev nodes of unwanted region of matrix spectrum.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param n Matrix dimension.
 *  @param iter Current Lanczos iteration.
 *  @param iter_new Lanczos iteration after restart.
 *  @param shiftUpper Pointer (host memory) to upper bound for unwanted
 *    region. Value is ignored if less than *shiftLower. If a
 *    stronger upper bound has been found, the value is updated on
 *    exit.
 *  @param shiftLower Pointer (host memory) to lower bound for unwanted
 *    region. Value is ignored if greater than *shiftUpper. If a
 *    stronger lower bound has been found, the value is updated on
 *    exit.
 *  @param alpha_host (Input/output, host memory, iter entries)
 *    Diagonal entries of Lanczos system.
 *  @param beta_host (Input/output, host memory, iter entries)
 *    Off-diagonal entries of Lanczos system.
 *  @param V_host (Output, host memory, iter*iter entries)
 *    Orthonormal transform used to obtain restarted system. Matrix
 *    dimensions are iter x iter.
 *  @param work_host (Output, host memory, 4*iter entries)
 *    Workspace.
 *  @param lanczosVecs_dev (Input/output, device memory, n*(iter+1)
 *    entries) Lanczos vectors. Vectors are stored as columns of a
 *    column-major matrix with dimensions n x (iter+1).
 *  @param work_dev (Output, device memory, (n+iter)*iter entries)
 *    Workspace.
 *  @param smallest_eig specifies whether smallest (true) or largest
 *    (false) eigenvalues are to be calculated.
 *  @return error flag.
 */
template <typename index_type_t, typename value_type_t>
static int lanczosRestart(raft::resources const& handle,
                          index_type_t n,
                          index_type_t iter,
                          index_type_t iter_new,
                          value_type_t* shiftUpper,
                          value_type_t* shiftLower,
                          value_type_t* __restrict__ alpha_host,
                          value_type_t* __restrict__ beta_host,
                          value_type_t* __restrict__ V_host,
                          value_type_t* __restrict__ work_host,
                          value_type_t* __restrict__ lanczosVecs_dev,
                          value_type_t* __restrict__ work_dev,
                          bool smallest_eig)
{
  // -------------------------------------------------------
  // Variable declaration
  // -------------------------------------------------------

  // Useful constants
  constexpr value_type_t zero = 0;
  constexpr value_type_t one  = 1;

  auto cublas_h = resource::get_cublas_handle(handle);
  auto stream   = resource::get_cuda_stream(handle);

  // Loop index
  index_type_t i;

  // Number of implicit restart steps
  //   Assumed to be even since each call to Francis algorithm is
  //   equivalent to two calls of QR algorithm
  index_type_t restartSteps = iter - iter_new;

  // Ritz values from Lanczos method
  value_type_t* ritzVals_host = work_host + 3 * iter;
  // Shifts for implicit restart
  value_type_t* shifts_host;

  // Orthonormal matrix for similarity transform
  value_type_t* V_dev = work_dev + n * iter;

  // -------------------------------------------------------
  // Implementation
  // -------------------------------------------------------

  // Compute Ritz values
  memcpy(ritzVals_host, alpha_host, iter * sizeof(value_type_t));
  memcpy(work_host, beta_host, (iter - 1) * sizeof(value_type_t));
  Lapack<value_type_t>::sterf(iter, ritzVals_host, work_host);

  // Debug: Print largest eigenvalues
  // for (int i = iter-iter_new; i < iter; ++i)
  //  std::cout <<*(ritzVals_host+i)<< " ";
  // std::cout <<std::endl;

  // Initialize similarity transform with identity matrix
  memset(V_host, 0, iter * iter * sizeof(value_type_t));
  for (i = 0; i < iter; ++i)
    V_host[IDX(i, i, iter)] = 1;

  // Determine interval to suppress eigenvalues
  if (smallest_eig) {
    if (*shiftLower > *shiftUpper) {
      *shiftUpper = ritzVals_host[iter - 1];
      *shiftLower = ritzVals_host[iter_new];
    } else {
      *shiftUpper = std::max(*shiftUpper, ritzVals_host[iter - 1]);
      *shiftLower = std::min(*shiftLower, ritzVals_host[iter_new]);
    }
  } else {
    if (*shiftLower > *shiftUpper) {
      *shiftUpper = ritzVals_host[iter - iter_new - 1];
      *shiftLower = ritzVals_host[0];
    } else {
      *shiftUpper = std::max(*shiftUpper, ritzVals_host[iter - iter_new - 1]);
      *shiftLower = std::min(*shiftLower, ritzVals_host[0]);
    }
  }

  // Calculate Chebyshev nodes as shifts
  shifts_host = ritzVals_host;
  for (i = 0; i < restartSteps; ++i) {
    shifts_host[i] = cos((i + 0.5) * static_cast<value_type_t>(M_PI) / restartSteps);
    shifts_host[i] *= 0.5 * ((*shiftUpper) - (*shiftLower));
    shifts_host[i] += 0.5 * ((*shiftUpper) + (*shiftLower));
  }

  // Apply Francis QR algorithm to implicitly restart Lanczos
  for (i = 0; i < restartSteps; i += 2)
    if (francisQRIteration(
          iter, shifts_host[i], shifts_host[i + 1], alpha_host, beta_host, V_host, work_host))
      WARNING("error in implicitly shifted QR algorithm");

  // Obtain new residual
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    V_dev, V_host, iter * iter * sizeof(value_type_t), cudaMemcpyHostToDevice, stream));

  beta_host[iter - 1] = beta_host[iter - 1] * V_host[IDX(iter - 1, iter_new - 1, iter)];
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(cublas_h,
                                                   CUBLAS_OP_N,
                                                   n,
                                                   iter,
                                                   beta_host + iter_new - 1,
                                                   lanczosVecs_dev,
                                                   n,
                                                   V_dev + IDX(0, iter_new, iter),
                                                   1,
                                                   beta_host + iter - 1,
                                                   lanczosVecs_dev + IDX(0, iter, n),
                                                   1,
                                                   stream));

  // Obtain new Lanczos vectors
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemm(cublas_h,
                                                   CUBLAS_OP_N,
                                                   CUBLAS_OP_N,
                                                   n,
                                                   iter_new,
                                                   iter,
                                                   &one,
                                                   lanczosVecs_dev,
                                                   n,
                                                   V_dev,
                                                   iter,
                                                   &zero,
                                                   work_dev,
                                                   n,
                                                   stream));

  RAFT_CUDA_TRY(cudaMemcpyAsync(lanczosVecs_dev,
                                work_dev,
                                n * iter_new * sizeof(value_type_t),
                                cudaMemcpyDeviceToDevice,
                                stream));

  // Normalize residual to obtain new Lanczos vector
  RAFT_CUDA_TRY(cudaMemcpyAsync(lanczosVecs_dev + IDX(0, iter_new, n),
                                lanczosVecs_dev + IDX(0, iter, n),
                                n * sizeof(value_type_t),
                                cudaMemcpyDeviceToDevice,
                                stream));

  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasnrm2(
    cublas_h, n, lanczosVecs_dev + IDX(0, iter_new, n), 1, beta_host + iter_new - 1, stream));

  auto h_beta = 1 / beta_host[iter_new - 1];
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasscal(
    cublas_h, n, &h_beta, lanczosVecs_dev + IDX(0, iter_new, n), 1, stream));

  return 0;
}

/**
 * @brief  Compute smallest eigenvectors of symmetric matrix
 *    Computes eigenvalues and eigenvectors that are least
 *    positive. If matrix is positive definite or positive
 *    semidefinite, the computed eigenvalues are smallest in
 *    magnitude.
 *    The largest eigenvalue is estimated by performing several
 *    Lanczos iterations. An implicitly restarted Lanczos method is
 *    then applied to A+s*I, where s is negative the largest
 *    eigenvalue.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param A Matrix.
 *  @param nEigVecs Number of eigenvectors to compute.
 *  @param maxIter Maximum number of Lanczos steps. Does not include
 *    Lanczos steps used to estimate largest eigenvalue.
 *  @param restartIter Maximum size of Lanczos system before
 *    performing an implicit restart. Should be at least 4.
 *  @param tol Convergence tolerance. Lanczos iteration will
 *    terminate when the residual norm is less than tol*theta, where
 *    theta is an estimate for the smallest unwanted eigenvalue
 *    (i.e. the (nEigVecs+1)th smallest eigenvalue).
 *  @param reorthogonalize Whether to reorthogonalize Lanczos
 *    vectors.
 *  @param effIter On exit, pointer to final size of Lanczos system.
 *  @param totalIter On exit, pointer to total number of Lanczos
 *    iterations performed. Does not include Lanczos steps used to
 *    estimate largest eigenvalue.
 *  @param shift On exit, pointer to matrix shift (estimate for
 *    largest eigenvalue).
 *  @param alpha_host (Output, host memory, restartIter entries)
 *    Diagonal entries of Lanczos system.
 *  @param beta_host (Output, host memory, restartIter entries)
 *    Off-diagonal entries of Lanczos system.
 *  @param lanczosVecs_dev (Output, device memory, n*(restartIter+1)
 *    entries) Lanczos vectors. Vectors are stored as columns of a
 *    column-major matrix with dimensions n x (restartIter+1).
 *  @param work_dev (Output, device memory,
 *    (n+restartIter)*restartIter entries) Workspace.
 *  @param eigVals_dev (Output, device memory, nEigVecs entries)
 *    Largest eigenvalues of matrix.
 *  @param eigVecs_dev (Output, device memory, n*nEigVecs entries)
 *    Eigenvectors corresponding to smallest eigenvalues of
 *    matrix. Vectors are stored as columns of a column-major matrix
 *    with dimensions n x nEigVecs.
 *  @param seed random seed.
 *  @return error flag.
 */
template <typename index_type_t, typename value_type_t>
int computeSmallestEigenvectors(
  raft::resources const& handle,
  spectral::matrix::sparse_matrix_t<index_type_t, value_type_t> const* A,
  index_type_t nEigVecs,
  index_type_t maxIter,
  index_type_t restartIter,
  value_type_t tol,
  int conv_n_iters,
  float conv_eps,
  bool reorthogonalize,
  index_type_t* effIter,
  index_type_t* totalIter,
  value_type_t* shift,
  value_type_t* __restrict__ alpha_host,
  value_type_t* __restrict__ beta_host,
  value_type_t* __restrict__ lanczosVecs_dev,
  value_type_t* __restrict__ work_dev,
  value_type_t* __restrict__ eigVals_dev,
  value_type_t* __restrict__ eigVecs_dev,
  unsigned long long seed)
{
  // std::cout << "computeSmallestEigenvectors" << std::endl;
  // Useful constants
  constexpr value_type_t one  = 1;
  constexpr value_type_t zero = 0;

  // Matrix dimension
  index_type_t n = A->nrows_;

  // Shift for implicit restart
  value_type_t shiftUpper;
  value_type_t shiftLower;

  // Lanczos iteration counters
  index_type_t maxIter_curr = restartIter;  // Maximum size of Lanczos system

  // Status flags
  int status;

  // Loop index
  index_type_t i;

  // Host memory
  value_type_t* Z_host;     // Eigenvectors in Lanczos basis
  value_type_t* work_host;  // Workspace

  // -------------------------------------------------------
  // Check that parameters are valid
  // -------------------------------------------------------
  RAFT_EXPECTS(nEigVecs > 0 && nEigVecs <= n, "Invalid number of eigenvectors.");
  RAFT_EXPECTS(restartIter > 0, "Invalid restartIter.");
  RAFT_EXPECTS(tol > 0, "Invalid tolerance.");
  RAFT_EXPECTS(maxIter >= nEigVecs, "Invalid maxIter.");
  RAFT_EXPECTS(restartIter >= nEigVecs, "Invalid restartIter.");

  auto cublas_h = resource::get_cublas_handle(handle);
  auto stream   = resource::get_cuda_stream(handle);

  // -------------------------------------------------------
  // Variable initialization
  // -------------------------------------------------------

  // Total number of Lanczos iterations
  *totalIter = 0;

  // Allocate host memory
  std::vector<value_type_t> Z_host_v(restartIter * restartIter);
  std::vector<value_type_t> work_host_v(4 * restartIter);

  Z_host    = Z_host_v.data();
  work_host = work_host_v.data();

  // Initialize cuBLAS
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST, stream));

  // -------------------------------------------------------
  // Compute largest eigenvalue to determine shift
  // -------------------------------------------------------

  // Random number generator
  curandGenerator_t randGen;
  // Initialize random number generator
  curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_PHILOX4_32_10);

  curandSetPseudoRandomGeneratorSeed(randGen, seed);

  std::cout << "seed " << seed << std::endl;
  raft::random::RngState rng(seed);
  if constexpr (std::is_same_v<value_type_t, float>) {
    raft::random::normal(handle, rng, lanczosVecs_dev, n + n % 2, zero, one);
  } else {
    spectral::matrix::vector_t<float> lanczosVecs_dev_float(handle, n * (restartIter + 1));
    raft::random::normal(handle, rng, lanczosVecs_dev_float.raw(), n + n % 2, 0.0F, 1.0F);
    std::vector<float> lanczosVecs_host_float(n * (restartIter + 1), 0);
    std::vector<double> lanczosVecs_host_double(n * (restartIter + 1), 0);
    raft::copy(lanczosVecs_host_float.data(), lanczosVecs_dev_float.raw(), n * (restartIter + 1), stream);
    for (uint64_t i = 0; i < lanczosVecs_host_float.size(); i++) {
      lanczosVecs_host_double[i] = static_cast<double>(lanczosVecs_host_float[i]);
    }
    raft::copy(lanczosVecs_dev, lanczosVecs_host_double.data(), n * (restartIter + 1), stream);
  }
  


  // Initialize initial Lanczos vector
  // curandGenerateNormalX(randGen, lanczosVecs_dev, n + n % 2, zero, one);
  // print_device_vector("lanczosVecs_dev", lanczosVecs_dev, n * (restartIter + 1), std::cout);
  value_type_t normQ1;
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublasnrm2(cublas_h, n, lanczosVecs_dev, 1, &normQ1, stream));

  // print_device_vector("lanczosVecs_dev", lanczosVecs_dev, n * (restartIter + 1), std::cout);
  
  auto h_val = 1 / normQ1;
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublasscal(cublas_h, n, &h_val, lanczosVecs_dev, 1, stream));

  // Obtain tridiagonal matrix with Lanczos
  *effIter = 0;
  *shift   = 0;
  // print_device_vector("lanczosVecs_dev", lanczosVecs_dev, n * (restartIter + 1), std::cout);
  // print_device_vector("alpha_host", lanczosVecs_dev, n * (restartIter + 1), std::cout);
  status   = performLanczosIteration<index_type_t, value_type_t>(handle,
                                                               A,
                                                               effIter,
                                                               maxIter_curr,
                                                               *shift,
                                                               0.0,
                                                               reorthogonalize,
                                                               alpha_host,
                                                               beta_host,
                                                               lanczosVecs_dev,
                                                               work_dev,
                                                               work_host,
                                                               Z_host,
                                                               eigVals_dev,
                                                               nEigVecs,
                                                               totalIter,
                                                               conv_n_iters,
                                                               conv_eps);
  if (status) WARNING("error in Lanczos iteration");
  *totalIter += *effIter;

  // // Determine largest eigenvalue

  // Lapack<value_type_t>::sterf(*effIter, alpha_host, beta_host);
  // *shift = -alpha_host[*effIter - 1];

  // // -------------------------------------------------------
  // // Compute eigenvectors of shifted matrix
  // // -------------------------------------------------------

  // // Obtain tridiagonal matrix with Lanczos
  // *effIter = 0;

  // status = performLanczosIteration<index_type_t, value_type_t>(handle,
  //                                                              A,
  //                                                              effIter,
  //                                                              maxIter_curr,
  //                                                              *shift,
  //                                                              0,
  //                                                              reorthogonalize,
  //                                                              alpha_host,
  //                                                              beta_host,
  //                                                              lanczosVecs_dev,
  //                                                              work_dev,
  //                                                              work_host,
  //                                                              Z_host,
  //                                                              eigVals_dev,
  //                                                              nEigVecs,
  //                                                              totalIter,
  //                                                              conv_n_iters,
  //                                                              conv_eps);
  // if (status) WARNING("error in Lanczos iteration");
  // *totalIter += *effIter;

  // Apply Lanczos method until convergence
  shiftLower = 1;
  shiftUpper = -1;
  // float conv_n_iters = 5;
  // float conv_eps = 0.001;
  // float prev_conv = 0;
  // value_type_t prev_conv = 0;
  // int orig_conv_n_iters = conv_n_iters;
  // std::vector<value_type_t> prev_conv(nEigVecs, 0);

  while (*totalIter < maxIter && beta_host[*effIter - 1] > tol * fabs(shiftLower) && conv_n_iters > 0 && status == 0) {
    // Determine number of restart steps
    // Number of steps must be even due to Francis algorithm
    index_type_t iter_new = nEigVecs + 1;
    if (restartIter - (maxIter - *totalIter) > nEigVecs + 1)
      iter_new = restartIter - (maxIter - *totalIter);
    if ((restartIter - iter_new) % 2) iter_new -= 1;
    if (iter_new == *effIter) break;

    // Implicit restart of Lanczos method
    status = lanczosRestart<index_type_t, value_type_t>(handle,
                                                        n,
                                                        *effIter,
                                                        iter_new,
                                                        &shiftUpper,
                                                        &shiftLower,
                                                        alpha_host,
                                                        beta_host,
                                                        Z_host,
                                                        work_host,
                                                        lanczosVecs_dev,
                                                        work_dev,
                                                        true);
    if (status) WARNING("error in Lanczos implicit restart");
    *effIter = iter_new;

    // // Solve tridiagonal system
    // memcpy(work_host + 2 * (*effIter), alpha_host, (*effIter) * sizeof(value_type_t));
    // memcpy(work_host + 3 * (*effIter), beta_host, (*effIter - 1) * sizeof(value_type_t));
    // Lapack<value_type_t>::steqr('I',
    //                             *effIter,
    //                             work_host + 2 * (*effIter),
    //                             work_host + 3 * (*effIter),
    //                             Z_host,
    //                             *effIter,
    //                             work_host);

    // // Obtain desired eigenvalues by applying shift
    // for (i = 0; i < *effIter; ++i)
    //   work_host[i + 2 * (*effIter)] -= *shift;
    // for (i = *effIter; i < nEigVecs; ++i)
    //   work_host[i + 2 * (*effIter)] = 0;

    // // Copy results to device memory
    // RAFT_CUDA_TRY(cudaMemcpyAsync(eigVals_dev,
    //                               work_host + 2 * (*effIter),
    //                               nEigVecs * sizeof(value_type_t),
    //                               cudaMemcpyHostToDevice,
    //                               stream));
    
    // print_device_vector("lanczos restart", eigVals_dev, nEigVecs, std::cout);
    // std::cout << *totalIter << " " << *effIter << " " << iter_new << " " << maxIter_curr << std::endl;

    // Check for convergence
    if (beta_host[*effIter - 1] <= tol * fabs(shiftLower)) break;

    // Proceed with Lanczos method
    // raft::copy(&prev_conv, &(eigVals_dev[nEigVecs - 1]), 1, stream);

    status = performLanczosIteration<index_type_t, value_type_t>(handle,
                                                                 A,
                                                                 effIter,
                                                                 maxIter_curr,
                                                                 *shift,
                                                                 tol * fabs(shiftLower),
                                                                 reorthogonalize,
                                                                 alpha_host,
                                                                 beta_host,
                                                                 lanczosVecs_dev,
                                                                 work_dev,
                                                                 work_host,
                                                                 Z_host,
                                                                 eigVals_dev,
                                                                 nEigVecs,
                                                                 totalIter,
                                                                 conv_n_iters,
                                                                 conv_eps);
    if (status) WARNING("error in Lanczos iteration");
    *totalIter += *effIter - iter_new;
    if (status == 1) break;

    // Solve tridiagonal system
    memcpy(work_host + 2 * (*effIter), alpha_host, (*effIter) * sizeof(value_type_t));
    memcpy(work_host + 3 * (*effIter), beta_host, (*effIter - 1) * sizeof(value_type_t));
    Lapack<value_type_t>::steqr('I',
                                *effIter,
                                work_host + 2 * (*effIter),
                                work_host + 3 * (*effIter),
                                Z_host,
                                *effIter,
                                work_host);

    // Obtain desired eigenvalues by applying shift
    for (i = 0; i < *effIter; ++i)
      work_host[i + 2 * (*effIter)] -= *shift;
    for (i = *effIter; i < nEigVecs; ++i)
      work_host[i + 2 * (*effIter)] = 0;

    // Copy results to device memory
    RAFT_CUDA_TRY(cudaMemcpyAsync(eigVals_dev,
                                  work_host + 2 * (*effIter),
                                  nEigVecs * sizeof(value_type_t),
                                  cudaMemcpyHostToDevice,
                                  stream));

    RAFT_CUDA_TRY(cudaMemcpyAsync(work_dev,
                                  Z_host,
                                  (*effIter) * nEigVecs * sizeof(value_type_t),
                                  cudaMemcpyHostToDevice,
                                  stream));

    std::ofstream out_file("output2.txt"); // Open a file for writing
  
    // Check if the file is open
    if (!out_file.is_open()) {
      std::cerr << "Failed to open output file!" << std::endl;
    }

    // std::cout << ((*effIter) * (nEigVecs - 1)) << std::endl;


    // print_device_vector("work_dev", work_dev, (*effIter) * nEigVecs, out_file);
    // raft::device_vector_view<value_type_t, int, raft::col_major> work_dev_view = raft::make_device_vector_view(work_dev, (*effIter) * nEigVecs);
    raft::host_matrix_view<value_type_t, uint32_t, raft::col_major> Z_host_view = raft::make_host_matrix_view<value_type_t, uint32_t, raft::col_major>(Z_host, (*effIter), nEigVecs);
    std::vector<value_type_t> Z_host_last_row(nEigVecs);
    for (int i = 0; i < nEigVecs; i++) {
      Z_host_last_row[i] = beta_host[*effIter - 1] * Z_host_view((*effIter) - 1, i);
    }
    float Znorm = 0;
    for (int i = 0; i < nEigVecs; i++) {
      Znorm += Z_host_last_row[i] * Z_host_last_row[i];
    }
    std::sqrt(Znorm);
    std::cout << (*totalIter + (*effIter - 1)) << " " << Znorm << " " << tol << " " << beta_host[*effIter - 1] << " " << std::endl;
    print_host_vector("Z_host_last_row", Z_host_last_row.data(), nEigVecs, out_file);
    print_host_vector("work_dev", Z_host, (*effIter) * nEigVecs, out_file);

    if (Znorm <= tol) {
      status = 1;
      break;
    }

    // Solve tridiagonal system
    // memcpy(work_host + 2 * (*effIter), alpha_host, (*effIter) * sizeof(value_type_t));
    // memcpy(work_host + 3 * (*effIter), beta_host, (*effIter - 1) * sizeof(value_type_t));
    // Lapack<value_type_t>::steqr('I',
    //                             *effIter,
    //                             work_host + 2 * (*effIter),
    //                             work_host + 3 * (*effIter),
    //                             Z_host,
    //                             *effIter,
    //                             work_host);

    // // Obtain desired eigenvalues by applying shift
    // for (i = 0; i < *effIter; ++i)
    //   work_host[i + 2 * (*effIter)] -= *shift;
    // for (i = *effIter; i < nEigVecs; ++i)
    //   work_host[i + 2 * (*effIter)] = 0;

    // // Copy results to device memory
    // RAFT_CUDA_TRY(cudaMemcpyAsync(eigVals_dev,
    //                               work_host + 2 * (*effIter),
    //                               nEigVecs * sizeof(value_type_t),
    //                               cudaMemcpyHostToDevice,
    //                               stream));

    // print_device_vector("lanczos iteration", eigVals_dev, nEigVecs, std::cout);
    // std::cout << *totalIter << " " << *effIter << " " << iter_new << " " << maxIter_curr << " "<< (beta_host[*effIter - 1] > tol * fabs(shiftLower)) << std::endl;

    // // value_type_t curr_conv;
    // std::vector<value_type_t> curr_conv(nEigVecs, 0);
    // raft::copy(curr_conv.data(), eigVals_dev, nEigVecs, stream);

    // for (i = 0; i < nEigVecs; i++) {
    //   if (fabs(curr_conv[i] - prev_conv[i]) > conv_eps) {
    //     conv_n_iters = orig_conv_n_iters;
    //     raft::copy(prev_conv.data(), eigVals_dev, nEigVecs, stream);
    //     break;
    //   }
    //   if (i == nEigVecs - 1) {
    //     conv_n_iters -= 1;
    //   }
    // }
  }

  // Warning if Lanczos has failed to converge
  // if (beta_host[*effIter - 1] > tol * fabs(shiftLower)) {
  //   std::cout << beta_host[*effIter - 1] << " failed to converge" << std::endl;
  //   WARNING("implicitly restarted Lanczos failed to converge");
  // } else {
  //   std::cout << beta_host[*effIter - 1] << " good" << std::endl;
  // }

  // Solve tridiagonal system
  memcpy(work_host + 2 * (*effIter), alpha_host, (*effIter) * sizeof(value_type_t));
  memcpy(work_host + 3 * (*effIter), beta_host, (*effIter - 1) * sizeof(value_type_t));
  Lapack<value_type_t>::steqr('I',
                              *effIter,
                              work_host + 2 * (*effIter),
                              work_host + 3 * (*effIter),
                              Z_host,
                              *effIter,
                              work_host);

  // Obtain desired eigenvalues by applying shift
  for (i = 0; i < *effIter; ++i)
    work_host[i + 2 * (*effIter)] -= *shift;
  for (i = *effIter; i < nEigVecs; ++i)
    work_host[i + 2 * (*effIter)] = 0;

  // Copy results to device memory
  RAFT_CUDA_TRY(cudaMemcpyAsync(eigVals_dev,
                                work_host + 2 * (*effIter),
                                nEigVecs * sizeof(value_type_t),
                                cudaMemcpyHostToDevice,
                                stream));

  RAFT_CUDA_TRY(cudaMemcpyAsync(work_dev,
                                Z_host,
                                (*effIter) * nEigVecs * sizeof(value_type_t),
                                cudaMemcpyHostToDevice,
                                stream));
  RAFT_CHECK_CUDA(stream);

  // Convert eigenvectors from Lanczos basis to standard basis
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemm(cublas_h,
                                                   CUBLAS_OP_N,
                                                   CUBLAS_OP_N,
                                                   n,
                                                   nEigVecs,
                                                   *effIter,
                                                   &one,
                                                   lanczosVecs_dev,
                                                   n,
                                                   work_dev,
                                                   *effIter,
                                                   &zero,
                                                   eigVecs_dev,
                                                   n,
                                                   stream));

  // Clean up and exit
  curandDestroyGenerator(randGen);
  return 0;
}

template <typename index_type_t, typename value_type_t>
void cupy_lanczos_fast(
  spectral::matrix::sparse_matrix_t<index_type_t, value_type_t> const* A,
  int ncv
)
{
  // int n = A->nrows_;

  
}

template <typename index_type_t, typename value_type_t>
void cupy_aux(
  raft::resources const& handle,
  spectral::matrix::sparse_matrix_t<index_type_t, value_type_t> const* A,
  raft::device_matrix_view<value_type_t, uint32_t, raft::row_major> V,
  raft::device_matrix_view<value_type_t> u,
  raft::device_matrix_view<value_type_t> alpha,
  raft::device_matrix_view<value_type_t> beta,
  int start_idx,
  int end_idx,
  int ncv,
  raft::device_matrix_view<value_type_t> v,
  raft::device_matrix_view<value_type_t> uu,
  raft::device_matrix_view<value_type_t> vv
)
{
  auto stream   = resource::get_cuda_stream(handle);

  int n = A->nrows_;

  // int b = 0;
  // int one = 1;
  // int zero = 0;
  // int mone = -1;

  // auto V_const = raft::make_device_matrix_view<const value_type_t, uint32_t, raft::row_major>(V.data_handle(), ncv, n);


  // v[...] = V[i_start]
  raft::copy(v.data_handle(), &(V(start_idx, 0)), n, stream);
  // auto mp = raft::make_device_vector<int, uint32_t>(handle, 1);
  // raft::matrix::fill(handle, mp.view(), start_idx);
  // auto mp_const = raft::make_device_vector_view<const int, uint32_t>(mp.data_handle(), 1);
  // auto v_view = raft::make_device_matrix_view<value_type_t, uint32_t, raft::row_major>(v.data_handle(), 1, n);

  // raft::matrix::gather<value_type_t, int, uint32_t>(handle, V_const, mp_const, v_view);


  print_device_vector("v[...]", v.data_handle(), n, std::cout);
}



template <typename index_type_t, typename value_type_t>
int cupy_smallest(
  raft::resources const& handle,
  spectral::matrix::sparse_matrix_t<index_type_t, value_type_t> const* A,
  index_type_t nEigVecs,
  index_type_t maxIter,
  index_type_t restartIter,
  value_type_t tol,
  int conv_n_iters,
  float conv_eps,
  bool reorthogonalize,
  index_type_t* effIter,
  index_type_t* totalIter,
  value_type_t* shift,
  value_type_t* __restrict__ alpha_host,
  value_type_t* __restrict__ beta_host,
  value_type_t* __restrict__ lanczosVecs_dev,
  value_type_t* __restrict__ work_dev,
  value_type_t* __restrict__ eigVals_dev,
  value_type_t* __restrict__ eigVecs_dev,
  value_type_t* __restrict__ v0,
  unsigned long long seed)
{
  std::cout << "hello cupy smallest " << A->nrows_ << " " << A->ncols_ << " " << A->nnz_ << std::endl;

  int n = A->nrows_;
  int ncv = restartIter;
  raft::print_device_vector("hello cupy v0 init", v0, n, std::cout);


  raft::device_matrix<value_type_t, uint32_t, raft::row_major> V = raft::make_device_matrix<value_type_t, uint32_t, raft::row_major>(handle, ncv, n);
  raft::device_matrix_view<value_type_t> u_view = raft::make_device_matrix_view<value_type_t>(V.data_handle(), 1, n); // First Row V[0]
  raft::device_matrix_view<const value_type_t> v0_view = raft::make_device_matrix_view<const value_type_t>(v0, 1, n);
  raft::linalg::row_normalize(handle, v0_view, u_view, raft::linalg::L2Norm);

  print_device_vector("V[0]", V.data_handle(), n, std::cout);

  raft::device_matrix<value_type_t, uint32_t, raft::row_major> alpha = raft::make_device_matrix<value_type_t, uint32_t, raft::row_major>(handle, 1, ncv);
  raft::device_matrix<value_type_t, uint32_t, raft::row_major> beta = raft::make_device_matrix<value_type_t, uint32_t, raft::row_major>(handle, 1, ncv);
  value_type_t zero = 0;
  raft::matrix::fill(handle, alpha.view(), zero);
  raft::matrix::fill(handle, beta.view(), zero);

  // start allocating for cupy_lanczos_fast()

  // cusparse_handle = None
  // if _csr.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
  //     cusparse_handle = device.get_cusparse_handle()
  //     spmv_op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
  //     spmv_alpha = numpy.array(1.0, A.dtype)
  //     spmv_beta = numpy.array(0.0, A.dtype)
  //     spmv_cuda_dtype = _dtype.to_cuda_dtype(A.dtype)
  //     spmv_alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT

  // v = cupy.empty((n,), dtype=A.dtype)
  // uu = cupy.empty((ncv,), dtype=A.dtype)
  // vv = cupy.empty((n,), dtype=A.dtype)
  // b = cupy.empty((), dtype=A.dtype)
  // one = numpy.array(1.0, dtype=A.dtype)
  // zero = numpy.array(0.0, dtype=A.dtype)
  // mone = numpy.array(-1.0, dtype=A.dtype)

  raft::device_matrix<value_type_t, uint32_t, raft::row_major> v = raft::make_device_matrix<value_type_t, uint32_t, raft::row_major>(handle, 1, n);
  raft::device_matrix<value_type_t, uint32_t, raft::row_major> uu = raft::make_device_matrix<value_type_t, uint32_t, raft::row_major>(handle, 1, ncv);
  raft::device_matrix<value_type_t, uint32_t, raft::row_major> vv = raft::make_device_matrix<value_type_t, uint32_t, raft::row_major>(handle, 1, n);

  // cupy_aux(A, V.view(), u_view, alpha.view(), beta.view());
  cupy_aux(handle, A, V.view(), u_view, alpha.view(), beta.view(), 0, ncv, ncv, v.view(), uu.view(), vv.view());

  return 0;
}









template <typename index_type_t, typename value_type_t>
int computeSmallestEigenvectors(
  raft::resources const& handle,
  spectral::matrix::sparse_matrix_t<index_type_t, value_type_t> const& A,
  index_type_t nEigVecs,
  index_type_t maxIter,
  index_type_t restartIter,
  value_type_t tol,
  int conv_n_iters,
  float conv_eps,
  bool reorthogonalize,
  index_type_t& iter,
  value_type_t* __restrict__ eigVals_dev,
  value_type_t* __restrict__ eigVecs_dev,
  value_type_t* __restrict__ v0,
  unsigned long long seed = 1234567)
{
  // Matrix dimension
  index_type_t n = A.nrows_;

  // Check that parameters are valid
  RAFT_EXPECTS(nEigVecs > 0 && nEigVecs <= n, "Invalid number of eigenvectors.");
  RAFT_EXPECTS(restartIter > 0, "Invalid restartIter.");
  RAFT_EXPECTS(tol > 0, "Invalid tolerance.");
  RAFT_EXPECTS(maxIter >= nEigVecs, "Invalid maxIter.");
  RAFT_EXPECTS(restartIter >= nEigVecs, "Invalid restartIter.");

  // Allocate memory
  std::vector<value_type_t> alpha_host_v(restartIter);
  std::vector<value_type_t> beta_host_v(restartIter);

  value_type_t* alpha_host = alpha_host_v.data();
  value_type_t* beta_host  = beta_host_v.data();

  spectral::matrix::vector_t<value_type_t> lanczosVecs_dev(handle, n * (restartIter + 1));
  spectral::matrix::vector_t<value_type_t> work_dev(handle, (n + restartIter) * restartIter);

  // Perform Lanczos method
  index_type_t effIter;
  value_type_t shift;
  int status = cupy_smallest(handle,
                            &A,
                            nEigVecs,
                            maxIter,
                            restartIter,
                            tol,
                            conv_n_iters,
                            conv_eps,
                            reorthogonalize,
                            &effIter,
                            &iter,
                            &shift,
                            alpha_host,
                            beta_host,
                            lanczosVecs_dev.raw(),
                            work_dev.raw(),
                            eigVals_dev,
                            eigVecs_dev,
                            v0,
                            seed);

  // Clean up and return
  return status;
}

/**
 *  @brief Compute largest eigenvectors of symmetric matrix
 *    Computes eigenvalues and eigenvectors that are least
 *    positive. If matrix is positive definite or positive
 *    semidefinite, the computed eigenvalues are largest in
 *    magnitude.
 *    The largest eigenvalue is estimated by performing several
 *    Lanczos iterations. An implicitly restarted Lanczos method is
 *    then applied.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param A Matrix.
 *  @param nEigVecs Number of eigenvectors to compute.
 *  @param maxIter Maximum number of Lanczos steps.
 *  @param restartIter Maximum size of Lanczos system before
 *    performing an implicit restart. Should be at least 4.
 *  @param tol Convergence tolerance. Lanczos iteration will
 *    terminate when the residual norm is less than tol*theta, where
 *    theta is an estimate for the largest unwanted eigenvalue
 *    (i.e. the (nEigVecs+1)th largest eigenvalue).
 *  @param reorthogonalize Whether to reorthogonalize Lanczos
 *    vectors.
 *  @param effIter On exit, pointer to final size of Lanczos system.
 *  @param totalIter On exit, pointer to total number of Lanczos
 *    iterations performed.
 *  @param alpha_host (Output, host memory, restartIter entries)
 *    Diagonal entries of Lanczos system.
 *  @param beta_host (Output, host memory, restartIter entries)
 *    Off-diagonal entries of Lanczos system.
 *  @param lanczosVecs_dev (Output, device memory, n*(restartIter+1)
 *    entries) Lanczos vectors. Vectors are stored as columns of a
 *    column-major matrix with dimensions n x (restartIter+1).
 *  @param work_dev (Output, device memory,
 *    (n+restartIter)*restartIter entries) Workspace.
 *  @param eigVals_dev (Output, device memory, nEigVecs entries)
 *    Largest eigenvalues of matrix.
 *  @param eigVecs_dev (Output, device memory, n*nEigVecs entries)
 *    Eigenvectors corresponding to largest eigenvalues of
 *    matrix. Vectors are stored as columns of a column-major matrix
 *    with dimensions n x nEigVecs.
 *  @param seed random seed.
 *  @return error flag.
 */
template <typename index_type_t, typename value_type_t>
int computeLargestEigenvectors(
  raft::resources const& handle,
  spectral::matrix::sparse_matrix_t<index_type_t, value_type_t> const* A,
  index_type_t nEigVecs,
  index_type_t maxIter,
  index_type_t restartIter,
  value_type_t tol,
  bool reorthogonalize,
  index_type_t* effIter,
  index_type_t* totalIter,
  value_type_t* __restrict__ alpha_host,
  value_type_t* __restrict__ beta_host,
  value_type_t* __restrict__ lanczosVecs_dev,
  value_type_t* __restrict__ work_dev,
  value_type_t* __restrict__ eigVals_dev,
  value_type_t* __restrict__ eigVecs_dev,
  unsigned long long seed)
{
  // Useful constants
  constexpr value_type_t one  = 1;
  constexpr value_type_t zero = 0;

  // Matrix dimension
  index_type_t n = A->nrows_;

  // Lanczos iteration counters
  index_type_t maxIter_curr = restartIter;  // Maximum size of Lanczos system

  // Status flags
  int status;

  // Loop index
  index_type_t i;

  // Host memory
  value_type_t* Z_host;     // Eigenvectors in Lanczos basis
  value_type_t* work_host;  // Workspace

  // -------------------------------------------------------
  // Check that LAPACK is enabled
  // -------------------------------------------------------
  // Lapack<value_type_t>::check_lapack_enabled();

  // -------------------------------------------------------
  // Check that parameters are valid
  // -------------------------------------------------------
  RAFT_EXPECTS(nEigVecs > 0 && nEigVecs <= n, "Invalid number of eigenvectors.");
  RAFT_EXPECTS(restartIter > 0, "Invalid restartIter.");
  RAFT_EXPECTS(tol > 0, "Invalid tolerance.");
  RAFT_EXPECTS(maxIter >= nEigVecs, "Invalid maxIter.");
  RAFT_EXPECTS(restartIter >= nEigVecs, "Invalid restartIter.");

  auto cublas_h = resource::get_cublas_handle(handle);
  auto stream   = resource::get_cuda_stream(handle);

  // -------------------------------------------------------
  // Variable initialization
  // -------------------------------------------------------

  // Total number of Lanczos iterations
  *totalIter = 0;

  // Allocate host memory
  std::vector<value_type_t> Z_host_v(restartIter * restartIter);
  std::vector<value_type_t> work_host_v(4 * restartIter);

  Z_host    = Z_host_v.data();
  work_host = work_host_v.data();

  // Initialize cuBLAS
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST, stream));

  // -------------------------------------------------------
  // Compute largest eigenvalue
  // -------------------------------------------------------

  // Random number generator
  curandGenerator_t randGen;
  // Initialize random number generator
  curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(randGen, seed);
  // Initialize initial Lanczos vector
  curandGenerateNormalX(randGen, lanczosVecs_dev, n + n % 2, zero, one);
  value_type_t normQ1;
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublasnrm2(cublas_h, n, lanczosVecs_dev, 1, &normQ1, stream));

  auto h_val = 1 / normQ1;
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublasscal(cublas_h, n, &h_val, lanczosVecs_dev, 1, stream));

  // Obtain tridiagonal matrix with Lanczos
  *effIter               = 0;
  value_type_t shift_val = 0.0;
  value_type_t* shift    = &shift_val;

  status = performLanczosIteration<index_type_t, value_type_t>(handle,
                                                               A,
                                                               effIter,
                                                               maxIter_curr,
                                                               *shift,
                                                               0,
                                                               reorthogonalize,
                                                               alpha_host,
                                                               beta_host,
                                                               lanczosVecs_dev,
                                                               work_dev,
                                                               work_host,
                                                               Z_host,
                                                               eigVals_dev,
                                                               nEigVecs,
                                                               totalIter,
                                                               1,
                                                               1);
  if (status) WARNING("error in Lanczos iteration");
  *totalIter += *effIter;

  // Apply Lanczos method until convergence
  value_type_t shiftLower = 1;
  value_type_t shiftUpper = -1;
  while (*totalIter < maxIter && beta_host[*effIter - 1] > tol * shiftLower) {
    // Determine number of restart steps
    //   Number of steps must be even due to Francis algorithm
    index_type_t iter_new = nEigVecs + 1;
    if (restartIter - (maxIter - *totalIter) > nEigVecs + 1)
      iter_new = restartIter - (maxIter - *totalIter);
    if ((restartIter - iter_new) % 2) iter_new -= 1;
    if (iter_new == *effIter) break;

    // Implicit restart of Lanczos method
    status = lanczosRestart<index_type_t, value_type_t>(handle,
                                                        n,
                                                        *effIter,
                                                        iter_new,
                                                        &shiftUpper,
                                                        &shiftLower,
                                                        alpha_host,
                                                        beta_host,
                                                        Z_host,
                                                        work_host,
                                                        lanczosVecs_dev,
                                                        work_dev,
                                                        false);
    if (status) WARNING("error in Lanczos implicit restart");
    *effIter = iter_new;

    // Check for convergence
    if (beta_host[*effIter - 1] <= tol * fabs(shiftLower)) break;

    // Proceed with Lanczos method

    status = performLanczosIteration<index_type_t, value_type_t>(handle,
                                                                 A,
                                                                 effIter,
                                                                 maxIter_curr,
                                                                 *shift,
                                                                 tol * fabs(shiftLower),
                                                                 reorthogonalize,
                                                                 alpha_host,
                                                                 beta_host,
                                                                 lanczosVecs_dev,
                                                                 work_dev,
                                                                 work_host,
                                                                 Z_host,
                                                                 eigVals_dev,
                                                                 nEigVecs,
                                                                 totalIter,
                                                                 1,
                                                                 1);
    if (status) WARNING("error in Lanczos iteration");
    *totalIter += *effIter - iter_new;
  }

  // Warning if Lanczos has failed to converge
  if (beta_host[*effIter - 1] > tol * fabs(shiftLower)) {
    WARNING("implicitly restarted Lanczos failed to converge");
  }
  for (int i = 0; i < restartIter; ++i) {
    for (int j = 0; j < restartIter; ++j)
      Z_host[i * restartIter + j] = 0;
  }
  // Solve tridiagonal system
  memcpy(work_host + 2 * (*effIter), alpha_host, (*effIter) * sizeof(value_type_t));
  memcpy(work_host + 3 * (*effIter), beta_host, (*effIter - 1) * sizeof(value_type_t));
  Lapack<value_type_t>::steqr('I',
                              *effIter,
                              work_host + 2 * (*effIter),
                              work_host + 3 * (*effIter),
                              Z_host,
                              *effIter,
                              work_host);

  // note: We need to pick the top nEigVecs eigenvalues
  // but effItter can be larger than nEigVecs
  // hence we add an offset for that case, because we want to access top nEigVecs eigenpairs in the
  // matrix of size effIter. remember the array is sorted, so it is not needed for smallest
  // eigenvalues case because the first ones are the smallest ones

  index_type_t top_eigenparis_idx_offset = *effIter - nEigVecs;

  // Debug : print nEigVecs largest eigenvalues
  // for (int i = top_eigenparis_idx_offset; i < *effIter; ++i)
  //  std::cout <<*(work_host+(2*(*effIter)+i))<< " ";
  // std::cout <<std::endl;

  // Debug : print nEigVecs largest eigenvectors
  // for (int i = top_eigenparis_idx_offset; i < *effIter; ++i)
  //{
  //  for (int j = 0; j < *effIter; ++j)
  //    std::cout <<Z_host[i*(*effIter)+j]<< " ";
  //  std::cout <<std::endl;
  //}

  // Obtain desired eigenvalues by applying shift
  for (i = 0; i < *effIter; ++i)
    work_host[i + 2 * (*effIter)] -= *shift;

  for (i = 0; i < top_eigenparis_idx_offset; ++i)
    work_host[i + 2 * (*effIter)] = 0;

  // Copy results to device memory
  // skip smallest eigenvalue if needed
  RAFT_CUDA_TRY(cudaMemcpyAsync(eigVals_dev,
                                work_host + 2 * (*effIter) + top_eigenparis_idx_offset,
                                nEigVecs * sizeof(value_type_t),
                                cudaMemcpyHostToDevice,
                                stream));

  // skip smallest eigenvector if needed
  RAFT_CUDA_TRY(cudaMemcpyAsync(work_dev,
                                Z_host + (top_eigenparis_idx_offset * (*effIter)),
                                (*effIter) * nEigVecs * sizeof(value_type_t),
                                cudaMemcpyHostToDevice,
                                stream));

  RAFT_CHECK_CUDA(stream);

  // Convert eigenvectors from Lanczos basis to standard basis
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemm(cublas_h,
                                                   CUBLAS_OP_N,
                                                   CUBLAS_OP_N,
                                                   n,
                                                   nEigVecs,
                                                   *effIter,
                                                   &one,
                                                   lanczosVecs_dev,
                                                   n,
                                                   work_dev,
                                                   *effIter,
                                                   &zero,
                                                   eigVecs_dev,
                                                   n,
                                                   stream));

  // Clean up and exit
  curandDestroyGenerator(randGen);
  return 0;
}

template <typename index_type_t, typename value_type_t>
int computeLargestEigenvectors(
  raft::resources const& handle,
  spectral::matrix::sparse_matrix_t<index_type_t, value_type_t> const& A,
  index_type_t nEigVecs,
  index_type_t maxIter,
  index_type_t restartIter,
  value_type_t tol,
  bool reorthogonalize,
  index_type_t& iter,
  value_type_t* __restrict__ eigVals_dev,
  value_type_t* __restrict__ eigVecs_dev,
  unsigned long long seed = 123456)
{
  // Matrix dimension
  index_type_t n = A.nrows_;

  // Check that parameters are valid
  RAFT_EXPECTS(nEigVecs > 0 && nEigVecs <= n, "Invalid number of eigenvectors.");
  RAFT_EXPECTS(restartIter > 0, "Invalid restartIter.");
  RAFT_EXPECTS(tol > 0, "Invalid tolerance.");
  RAFT_EXPECTS(maxIter >= nEigVecs, "Invalid maxIter.");
  RAFT_EXPECTS(restartIter >= nEigVecs, "Invalid restartIter.");

  // Allocate memory
  std::vector<value_type_t> alpha_host_v(restartIter);
  std::vector<value_type_t> beta_host_v(restartIter);

  value_type_t* alpha_host = alpha_host_v.data();
  value_type_t* beta_host  = beta_host_v.data();

  spectral::matrix::vector_t<value_type_t> lanczosVecs_dev(handle, n * (restartIter + 1));
  spectral::matrix::vector_t<value_type_t> work_dev(handle, (n + restartIter) * restartIter);

  // Perform Lanczos method
  index_type_t effIter;
  int status = computeLargestEigenvectors(handle,
                                          &A,
                                          nEigVecs,
                                          maxIter,
                                          restartIter,
                                          tol,
                                          reorthogonalize,
                                          &effIter,
                                          &iter,
                                          alpha_host,
                                          beta_host,
                                          lanczosVecs_dev.raw(),
                                          work_dev.raw(),
                                          eigVals_dev,
                                          eigVecs_dev,
                                          seed);

  // Clean up and return
  return status;
}

}  // namespace raft::sparse::solver::detail
