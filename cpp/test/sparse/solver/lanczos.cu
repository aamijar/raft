/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "../../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/random/rmat_rectangular_generator.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/degree.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/op/reduce.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/spectral/eigen_solvers.cuh>
#include <raft/spectral/matrix_wrappers.hpp>
#include <raft/util/cudart_utils.hpp>

#include <driver_types.h>

#include <gtest/gtest.h>
#include <sys/types.h>
#include <test_utils.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace raft {
namespace sparse {

template <typename index_type, typename value_type>
struct lanczos_inputs {
  int n_components;
  int restartiter;
  int maxiter;
  int conv_n_iters;
  float conv_eps;
  float tol;
  uint64_t seed;
  std::vector<index_type> rows;  // indptr
  std::vector<index_type> cols;  // indices
  std::vector<value_type> vals;  // data
  std::vector<value_type> expected_eigenvalues;
};

template <typename index_type, typename value_type>
struct rmat_lanczos_inputs {
  int n_components;
  int restartiter;
  int maxiter;
  int conv_n_iters;
  float conv_eps;
  float tol;
  uint64_t seed;
  int r_scale;
  int c_scale;
  float sparsity;
  std::vector<value_type> expected_eigenvalues;
};

template <typename index_type, typename value_type>
class dummy_lanczos_tests
  : public ::testing::TestWithParam<lanczos_inputs<index_type, value_type>> {};

template <typename index_type, typename value_type>
class rmat_lanczos_tests
  : public ::testing::TestWithParam<rmat_lanczos_inputs<index_type, value_type>> {
 public:
  rmat_lanczos_tests()
    : params(::testing::TestWithParam<rmat_lanczos_inputs<index_type, value_type>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      rng(params.seed),
      expected_eigenvalues(raft::make_device_vector<value_type, uint32_t, raft::col_major>(
        handle, params.n_components)),
      r_scale(params.r_scale),
      c_scale(params.c_scale),
      sparsity(params.sparsity)
  {
  }

 protected:
  void SetUp() override
  {
    raft::copy(expected_eigenvalues.data_handle(),
               params.expected_eigenvalues.data(),
               params.n_components,
               stream);
  }

  void TearDown() override {}

  void Run()
  {
    uint64_t n_edges   = sparsity * ((long long)(1 << r_scale) * (long long)(1 << c_scale));
    uint64_t n_nodes   = 1 << std::max(r_scale, c_scale);
    uint64_t theta_len = std::max(r_scale, c_scale) * 4;

    raft::device_vector<value_type, uint32_t, raft::row_major> theta =
      raft::make_device_vector<value_type, uint32_t, raft::row_major>(handle, theta_len);
    raft::random::uniform<value_type>(handle, rng, theta.view(), 0, 1);

    raft::device_matrix<index_type, uint32_t, raft::row_major> out =
      raft::make_device_matrix<index_type, uint32_t, raft::row_major>(handle, n_edges * 2, 2);
    raft::device_vector<index_type, uint32_t, raft::row_major> out_src =
      raft::make_device_vector<index_type, uint32_t, raft::row_major>(handle, n_edges);
    raft::device_vector<index_type, uint32_t, raft::row_major> out_dst =
      raft::make_device_vector<index_type, uint32_t, raft::row_major>(handle, n_edges);

    raft::random::RngState rng1{params.seed};

    raft::random::rmat_rectangular_gen<index_type, value_type>(handle,
                                                               rng1,
                                                               make_const_mdspan(theta.view()),
                                                               out.view(),
                                                               out_src.view(),
                                                               out_dst.view(),
                                                               r_scale,
                                                               c_scale);

    raft::device_vector<value_type, uint32_t, raft::row_major> out_data =
      raft::make_device_vector<value_type, uint32_t, raft::row_major>(handle, n_edges);
    raft::matrix::fill<value_type>(handle, out_data.view(), 1.0);
    raft::sparse::COO<value_type, index_type> coo(stream);

    raft::sparse::op::coo_sort(n_nodes,
                               n_nodes,
                               n_edges,
                               out_src.data_handle(),
                               out_dst.data_handle(),
                               out_data.data_handle(),
                               stream);
    raft::sparse::op::max_duplicates<index_type, value_type>(handle,
                                                             coo,
                                                             out_src.data_handle(),
                                                             out_dst.data_handle(),
                                                             out_data.data_handle(),
                                                             n_edges,
                                                             n_nodes,
                                                             n_nodes);

    raft::sparse::COO<value_type, index_type> symmetric_coo(stream);
    raft::sparse::linalg::symmetrize(
      handle, coo.rows(), coo.cols(), coo.vals(), coo.n_rows, coo.n_cols, coo.nnz, symmetric_coo);

    raft::device_vector<index_type, uint32_t, raft::row_major> row_indices =
      raft::make_device_vector<index_type, uint32_t, raft::row_major>(handle,
                                                                      symmetric_coo.n_rows + 1);
    raft::sparse::convert::sorted_coo_to_csr(symmetric_coo.rows(),
                                             symmetric_coo.nnz,
                                             row_indices.data_handle(),
                                             symmetric_coo.n_rows + 1,
                                             stream);

    int n_components = params.n_components;

    raft::device_vector<value_type, uint32_t, raft::row_major> v0 =
      raft::make_device_vector<value_type, uint32_t, raft::row_major>(handle, symmetric_coo.n_rows);

    raft::random::uniform<value_type>(handle, rng, v0.view(), 0, 1);
    // raft::spectral::matrix::sparse_matrix_t<index_type, value_type> const csr_m{handle,
    // row_indices.data_handle(), symmetric_coo.cols(), symmetric_coo.vals(), symmetric_coo.n_rows,
    // symmetric_coo.nnz}; raft::spectral::eigen_solver_config_t<index_type, value_type>
    // cfg{n_components, params.maxiter, params.restartiter, params.tol, false, rng.seed};
    std::tuple<index_type, value_type, index_type> stats;
    // raft::spectral::lanczos_solver_t<index_type, value_type> eigen_solver{cfg};

    raft::device_vector<value_type, uint32_t, raft::col_major> eigenvalues =
      raft::make_device_vector<value_type, uint32_t, raft::col_major>(handle, n_components);
    raft::device_matrix<value_type, uint32_t, raft::col_major> eigenvectors =
      raft::make_device_matrix<value_type, uint32_t, raft::col_major>(
        handle, symmetric_coo.n_rows, n_components);

    raft::spectral::matrix::sparse_matrix_t<index_type, value_type> const csr_m{
      handle,
      row_indices.data_handle(),
      symmetric_coo.cols(),
      symmetric_coo.vals(),
      symmetric_coo.n_rows,
      symmetric_coo.nnz};
    raft::sparse::solver::lanczos_solver_config<index_type, value_type> config{
      n_components, params.maxiter, params.restartiter, params.tol, rng.seed};
    std::get<0>(stats) =
      raft::sparse::solver::lanczos_compute_smallest_eigenvectors<index_type, value_type>(
        handle, csr_m, config, v0.view(), eigenvalues.view(), eigenvectors.view());

    // std::get<0>(stats) = eigen_solver.solve_smallest_eigenvectors(handle, csr_m,
    // eigenvalues.data_handle(), eigenvectors.data_handle(), v0.data_handle());

    ASSERT_TRUE(raft::devArrMatch<value_type>(eigenvalues.data_handle(),
                                              expected_eigenvalues.data_handle(),
                                              n_components,
                                              raft::CompareApprox<value_type>(1e-5),
                                              stream));
  }

 protected:
  rmat_lanczos_inputs<index_type, value_type> params;
  raft::resources handle;
  cudaStream_t stream;
  raft::random::RngState rng;
  int r_scale;
  int c_scale;
  float sparsity;
  raft::device_vector<value_type, uint32_t, raft::col_major> expected_eigenvalues;
};

template <typename index_type, typename value_type>
class lanczos_tests : public ::testing::TestWithParam<lanczos_inputs<index_type, value_type>> {
 public:
  lanczos_tests()
    : params(::testing::TestWithParam<lanczos_inputs<index_type, value_type>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      n(params.rows.size() - 1),
      nnz(params.vals.size()),
      rng(params.seed),
      rows(raft::make_device_vector<index_type, uint32_t, raft::row_major>(handle, n + 1)),
      cols(raft::make_device_vector<index_type, uint32_t, raft::row_major>(handle, nnz)),
      vals(raft::make_device_vector<value_type, uint32_t, raft::row_major>(handle, nnz)),
      v0(raft::make_device_vector<value_type, uint32_t, raft::row_major>(handle, n)),
      eigenvalues(raft::make_device_vector<value_type, uint32_t, raft::col_major>(
        handle, params.n_components)),
      eigenvectors(raft::make_device_matrix<value_type, uint32_t, raft::col_major>(
        handle, n, params.n_components)),
      expected_eigenvalues(raft::make_device_vector<value_type, uint32_t, raft::col_major>(
        handle, params.n_components))
  {
  }

 protected:
  void SetUp() override
  {
    raft::copy(rows.data_handle(), params.rows.data(), n + 1, stream);
    raft::copy(cols.data_handle(), params.cols.data(), nnz, stream);
    raft::copy(vals.data_handle(), params.vals.data(), nnz, stream);
    raft::copy(expected_eigenvalues.data_handle(),
               params.expected_eigenvalues.data(),
               params.n_components,
               stream);
  }

  void TearDown() override {}

  void Run()
  {
    raft::random::uniform<value_type>(handle, rng, v0.view(), 0, 1);
    // raft::spectral::matrix::sparse_matrix_t<index_type, value_type> const csr_m{handle,
    // rows.data_handle(), cols.data_handle(), vals.data_handle(), n, nnz};
    // raft::spectral::eigen_solver_config_t<index_type, value_type> cfg{params.n_components,
    // params.maxiter, params.restartiter, params.tol, false, params.seed};
    std::tuple<index_type, value_type, index_type> stats;
    // raft::spectral::lanczos_solver_t<index_type, value_type> eigen_solver{cfg};

    raft::spectral::matrix::sparse_matrix_t<index_type, value_type> const csr_m{
      handle, rows.data_handle(), cols.data_handle(), vals.data_handle(), n, nnz};
    raft::sparse::solver::lanczos_solver_config<index_type, value_type> config{
      params.n_components, params.maxiter, params.restartiter, params.tol, rng.seed};
    std::get<0>(stats) =
      raft::sparse::solver::lanczos_compute_smallest_eigenvectors<index_type, value_type>(
        handle, csr_m, config, v0.view(), eigenvalues.view(), eigenvectors.view());

    // std::get<0>(stats) = eigen_solver.solve_smallest_eigenvectors(handle, csr_m,
    // eigenvalues.data_handle(), eigenvectors.data_handle(), v0.data_handle());

    ASSERT_TRUE(raft::devArrMatch<value_type>(eigenvalues.data_handle(),
                                              expected_eigenvalues.data_handle(),
                                              params.n_components,
                                              raft::CompareApprox<value_type>(1e-5),
                                              stream));
  }

 protected:
  lanczos_inputs<index_type, value_type> params;
  raft::resources handle;
  cudaStream_t stream;
  int n;
  int nnz;
  raft::random::RngState rng;
  raft::device_vector<index_type, uint32_t, raft::row_major> rows;
  raft::device_vector<index_type, uint32_t, raft::row_major> cols;
  raft::device_vector<value_type, uint32_t, raft::row_major> vals;
  raft::device_vector<value_type, uint32_t, raft::row_major> v0;
  raft::device_vector<value_type, uint32_t, raft::col_major> eigenvalues;
  raft::device_matrix<value_type, uint32_t, raft::col_major> eigenvectors;
  raft::device_vector<value_type, uint32_t, raft::col_major> expected_eigenvalues;
};

const std::vector<lanczos_inputs<int, float>> inputsf = {
  {2,
   34,
   10000,
   0,
   0,
   1e-15,
   42,
   {0,   0,   0,   0,   3,   5,   6,   8,   9,   11,  16,  16,  18,  20,  23,  24,  27,
    30,  31,  33,  37,  37,  39,  41,  43,  44,  46,  46,  47,  49,  50,  50,  51,  53,
    57,  58,  59,  66,  67,  68,  69,  71,  72,  75,  78,  83,  86,  90,  93,  94,  96,
    98,  99,  101, 101, 104, 106, 108, 109, 109, 109, 109, 111, 113, 118, 120, 121, 123,
    124, 128, 132, 134, 136, 138, 139, 141, 145, 148, 151, 152, 154, 155, 157, 160, 164,
    167, 170, 170, 170, 173, 178, 179, 182, 184, 186, 191, 192, 196, 198, 198, 198},
   {44, 68, 74, 16, 36, 85, 34, 75, 61, 51, 83, 15, 33, 55, 69, 71, 18, 84, 70, 95, 71, 83,
    97, 83, 9,  36, 54, 4,  42, 46, 52, 11, 89, 31, 37, 74, 96, 36, 88, 56, 64, 68, 94, 82,
    35, 90, 50, 82, 85, 83, 19, 47, 94, 9,  44, 56, 79, 6,  25, 4,  15, 21, 52, 75, 79, 92,
    19, 72, 94, 94, 96, 80, 16, 54, 89, 46, 48, 63, 3,  33, 67, 73, 77, 46, 47, 75, 16, 43,
    45, 81, 32, 45, 68, 43, 55, 63, 27, 89, 8,  17, 36, 15, 42, 96, 9,  49, 22, 33, 77, 7,
    75, 78, 88, 43, 49, 66, 76, 91, 22, 82, 69, 63, 84, 44, 3,  23, 47, 81, 9,  65, 76, 92,
    12, 96, 9,  13, 38, 93, 44, 3,  19, 6,  36, 45, 61, 63, 69, 89, 44, 57, 94, 62, 33, 36,
    41, 46, 68, 24, 28, 64, 8,  13, 14, 29, 11, 66, 88, 5,  28, 93, 21, 62, 84, 18, 42, 50,
    76, 91, 25, 63, 89, 97, 36, 69, 72, 85, 23, 32, 39, 40, 77, 12, 19, 40, 54, 70, 13, 91},
   {0.4734894, 0.1402491, 0.7686475, 0.0416142, 0.2559651, 0.9360436, 0.7486080, 0.5206724,
    0.0374126, 0.8082515, 0.5993828, 0.4866583, 0.8907925, 0.9251201, 0.8566143, 0.9528994,
    0.4557763, 0.4907070, 0.4158074, 0.8311127, 0.9026024, 0.3103237, 0.5876446, 0.7585195,
    0.4866583, 0.4493615, 0.5909155, 0.0416142, 0.0963910, 0.6722401, 0.3468698, 0.4557763,
    0.1445242, 0.7720124, 0.9923756, 0.1227579, 0.7194629, 0.8916773, 0.4320931, 0.5840980,
    0.0216121, 0.3709223, 0.1705930, 0.8297898, 0.2409706, 0.9585592, 0.3171389, 0.0228039,
    0.4350971, 0.4939908, 0.7720124, 0.2722416, 0.1792683, 0.8907925, 0.1085757, 0.8745620,
    0.3298612, 0.7486080, 0.2409706, 0.2559651, 0.4493615, 0.8916773, 0.5540361, 0.5150571,
    0.9160119, 0.1767728, 0.9923756, 0.5717281, 0.1077409, 0.9368132, 0.6273088, 0.6616613,
    0.0963910, 0.9378265, 0.3059566, 0.3159291, 0.0449106, 0.9085807, 0.4734894, 0.1085757,
    0.2909013, 0.7787509, 0.7168902, 0.9691764, 0.2669757, 0.4389115, 0.6722401, 0.3159291,
    0.9691764, 0.7467896, 0.2722416, 0.2669757, 0.1532843, 0.0449106, 0.2023634, 0.8934466,
    0.3171389, 0.6594226, 0.8082515, 0.3468698, 0.5540361, 0.5909155, 0.9378265, 0.2909178,
    0.9251201, 0.2023634, 0.5840980, 0.8745620, 0.2624605, 0.0374126, 0.1034030, 0.3736577,
    0.3315690, 0.9085807, 0.8934466, 0.5548525, 0.2302140, 0.7827352, 0.0216121, 0.8262919,
    0.1646078, 0.5548525, 0.2658700, 0.2909013, 0.1402491, 0.3709223, 0.1532843, 0.5792196,
    0.8566143, 0.1646078, 0.0827300, 0.5810611, 0.4158074, 0.5188584, 0.9528994, 0.9026024,
    0.5717281, 0.7269946, 0.7787509, 0.7686475, 0.1227579, 0.5206724, 0.5150571, 0.4389115,
    0.1034030, 0.2302140, 0.0827300, 0.8961608, 0.7168902, 0.2624605, 0.4823034, 0.3736577,
    0.3298612, 0.9160119, 0.6616613, 0.7467896, 0.5792196, 0.8297898, 0.0228039, 0.8262919,
    0.5993828, 0.3103237, 0.7585195, 0.4939908, 0.4907070, 0.2658700, 0.0844443, 0.9360436,
    0.4350971, 0.6997072, 0.4320931, 0.3315690, 0.0844443, 0.1445242, 0.3059566, 0.6594226,
    0.8961608, 0.6498466, 0.9585592, 0.7827352, 0.6498466, 0.2812338, 0.1767728, 0.5810611,
    0.7269946, 0.6997072, 0.1705930, 0.1792683, 0.1077409, 0.9368132, 0.4823034, 0.8311127,
    0.7194629, 0.6273088, 0.2909178, 0.5188584, 0.5876446, 0.2812338},
   {-2.0369630, -1.7673520}}};

const std::vector<lanczos_inputs<int, double>> inputsd = {
  {2,
   34,
   10000,
   0,
   0,
   1e-15,
   42,
   {0,   0,   0,   0,   3,   5,   6,   8,   9,   11,  16,  16,  18,  20,  23,  24,  27,
    30,  31,  33,  37,  37,  39,  41,  43,  44,  46,  46,  47,  49,  50,  50,  51,  53,
    57,  58,  59,  66,  67,  68,  69,  71,  72,  75,  78,  83,  86,  90,  93,  94,  96,
    98,  99,  101, 101, 104, 106, 108, 109, 109, 109, 109, 111, 113, 118, 120, 121, 123,
    124, 128, 132, 134, 136, 138, 139, 141, 145, 148, 151, 152, 154, 155, 157, 160, 164,
    167, 170, 170, 170, 173, 178, 179, 182, 184, 186, 191, 192, 196, 198, 198, 198},
   {44, 68, 74, 16, 36, 85, 34, 75, 61, 51, 83, 15, 33, 55, 69, 71, 18, 84, 70, 95, 71, 83,
    97, 83, 9,  36, 54, 4,  42, 46, 52, 11, 89, 31, 37, 74, 96, 36, 88, 56, 64, 68, 94, 82,
    35, 90, 50, 82, 85, 83, 19, 47, 94, 9,  44, 56, 79, 6,  25, 4,  15, 21, 52, 75, 79, 92,
    19, 72, 94, 94, 96, 80, 16, 54, 89, 46, 48, 63, 3,  33, 67, 73, 77, 46, 47, 75, 16, 43,
    45, 81, 32, 45, 68, 43, 55, 63, 27, 89, 8,  17, 36, 15, 42, 96, 9,  49, 22, 33, 77, 7,
    75, 78, 88, 43, 49, 66, 76, 91, 22, 82, 69, 63, 84, 44, 3,  23, 47, 81, 9,  65, 76, 92,
    12, 96, 9,  13, 38, 93, 44, 3,  19, 6,  36, 45, 61, 63, 69, 89, 44, 57, 94, 62, 33, 36,
    41, 46, 68, 24, 28, 64, 8,  13, 14, 29, 11, 66, 88, 5,  28, 93, 21, 62, 84, 18, 42, 50,
    76, 91, 25, 63, 89, 97, 36, 69, 72, 85, 23, 32, 39, 40, 77, 12, 19, 40, 54, 70, 13, 91},
   {0.4734894, 0.1402491, 0.7686475, 0.0416142, 0.2559651, 0.9360436, 0.7486080, 0.5206724,
    0.0374126, 0.8082515, 0.5993828, 0.4866583, 0.8907925, 0.9251201, 0.8566143, 0.9528994,
    0.4557763, 0.4907070, 0.4158074, 0.8311127, 0.9026024, 0.3103237, 0.5876446, 0.7585195,
    0.4866583, 0.4493615, 0.5909155, 0.0416142, 0.0963910, 0.6722401, 0.3468698, 0.4557763,
    0.1445242, 0.7720124, 0.9923756, 0.1227579, 0.7194629, 0.8916773, 0.4320931, 0.5840980,
    0.0216121, 0.3709223, 0.1705930, 0.8297898, 0.2409706, 0.9585592, 0.3171389, 0.0228039,
    0.4350971, 0.4939908, 0.7720124, 0.2722416, 0.1792683, 0.8907925, 0.1085757, 0.8745620,
    0.3298612, 0.7486080, 0.2409706, 0.2559651, 0.4493615, 0.8916773, 0.5540361, 0.5150571,
    0.9160119, 0.1767728, 0.9923756, 0.5717281, 0.1077409, 0.9368132, 0.6273088, 0.6616613,
    0.0963910, 0.9378265, 0.3059566, 0.3159291, 0.0449106, 0.9085807, 0.4734894, 0.1085757,
    0.2909013, 0.7787509, 0.7168902, 0.9691764, 0.2669757, 0.4389115, 0.6722401, 0.3159291,
    0.9691764, 0.7467896, 0.2722416, 0.2669757, 0.1532843, 0.0449106, 0.2023634, 0.8934466,
    0.3171389, 0.6594226, 0.8082515, 0.3468698, 0.5540361, 0.5909155, 0.9378265, 0.2909178,
    0.9251201, 0.2023634, 0.5840980, 0.8745620, 0.2624605, 0.0374126, 0.1034030, 0.3736577,
    0.3315690, 0.9085807, 0.8934466, 0.5548525, 0.2302140, 0.7827352, 0.0216121, 0.8262919,
    0.1646078, 0.5548525, 0.2658700, 0.2909013, 0.1402491, 0.3709223, 0.1532843, 0.5792196,
    0.8566143, 0.1646078, 0.0827300, 0.5810611, 0.4158074, 0.5188584, 0.9528994, 0.9026024,
    0.5717281, 0.7269946, 0.7787509, 0.7686475, 0.1227579, 0.5206724, 0.5150571, 0.4389115,
    0.1034030, 0.2302140, 0.0827300, 0.8961608, 0.7168902, 0.2624605, 0.4823034, 0.3736577,
    0.3298612, 0.9160119, 0.6616613, 0.7467896, 0.5792196, 0.8297898, 0.0228039, 0.8262919,
    0.5993828, 0.3103237, 0.7585195, 0.4939908, 0.4907070, 0.2658700, 0.0844443, 0.9360436,
    0.4350971, 0.6997072, 0.4320931, 0.3315690, 0.0844443, 0.1445242, 0.3059566, 0.6594226,
    0.8961608, 0.6498466, 0.9585592, 0.7827352, 0.6498466, 0.2812338, 0.1767728, 0.5810611,
    0.7269946, 0.6997072, 0.1705930, 0.1792683, 0.1077409, 0.9368132, 0.4823034, 0.8311127,
    0.7194629, 0.6273088, 0.2909178, 0.5188584, 0.5876446, 0.2812338},
   {-2.0369630, -1.7673520}}};

const std::vector<rmat_lanczos_inputs<int, float>> rmat_inputsf = {
  {50, 100, 10000, 0, 0, 1e-9, 42, 12, 12, 1, {-122.526794, -74.00686,  -59.698284,  -54.68617,
                                               -49.686813,  -34.02644,  -32.130703,  -31.26906,
                                               -30.32097,   -22.946098, -20.497862,  -20.23817,
                                               -19.269697,  -18.42496,  -17.675667,  -17.013401,
                                               -16.734581,  -15.820215, -15.73925,   -15.448187,
                                               -15.044634,  -14.692028, -14.127425,  -13.967386,
                                               -13.6237755, -13.469393, -13.181225,  -12.777589,
                                               -12.623185,  -12.55508,  -12.2874565, -12.053391,
                                               -11.677346,  -11.558279, -11.163732,  -10.922034,
                                               -10.7936945, -10.558049, -10.205776,  -10.005316,
                                               -9.559181,   -9.491834,  -9.242631,   -8.883637,
                                               -8.765364,   -8.688508,  -8.458255,   -8.385196,
                                               -8.217982,   -8.0442095}}};

const std::vector<rmat_lanczos_inputs<int, float>> rmat_inputs_edge_case = {
  {100,
   300,
   10000,
   0,
   0,
   1e-9,
   42,
   12,
   12,
   1,
   {-1.22526756e+02, -7.40069504e+01, -5.96983109e+01, -5.46862068e+01, -4.96868439e+01,
    -3.40264435e+01, -3.21306839e+01, -3.12690392e+01, -3.03210258e+01, -2.29461250e+01,
    -2.04978676e+01, -2.02381744e+01, -1.92697086e+01, -1.84249725e+01, -1.76756725e+01,
    -1.70134144e+01, -1.67345791e+01, -1.58202209e+01, -1.57392349e+01, -1.54481869e+01,
    -1.50446243e+01, -1.46920280e+01, -1.41274376e+01, -1.39673843e+01, -1.36237764e+01,
    -1.34693928e+01, -1.31812143e+01, -1.27775812e+01, -1.26231880e+01, -1.25550766e+01,
    -1.22874584e+01, -1.20533924e+01, -1.16773510e+01, -1.15582829e+01, -1.11637363e+01,
    -1.09220333e+01, -1.07936945e+01, -1.05580463e+01, -1.02057772e+01, -1.00053129e+01,
    -9.55917740e+00, -9.49183655e+00, -9.24262238e+00, -8.88363647e+00, -8.76536846e+00,
    -8.68850899e+00, -8.45825481e+00, -8.38520622e+00, -8.21798038e+00, -8.04420948e+00,
    -7.90373087e+00, -7.83332729e+00, -7.54670286e+00, -7.50262451e+00, -7.36070538e+00,
    -7.06634855e+00, -6.89205170e+00, -6.64973640e+00, -6.46234751e+00, -5.98167992e+00,
    -5.67716694e+00, -5.48805237e+00, -5.00374651e+00, -4.64848948e+00, -6.70900226e-06,
    -5.04503123e-06, -1.94547101e-06, -5.66026663e-13, -5.23560958e-13, -4.79860509e-13,
    -4.48999019e-13, -4.35402040e-13, -4.26073429e-13, -4.10326368e-13, -4.09151066e-13,
    -3.81928457e-13, -3.71661062e-13, -3.63793847e-13, -3.51424022e-13, -3.45496228e-13,
    -3.36190629e-13, -3.27994251e-13, -3.12900720e-13, -3.00004786e-13, -2.84064601e-13,
    -2.75522199e-13, -2.58613199e-13, -2.47531948e-13, -2.35822267e-13, -2.04967106e-13,
    -1.92008627e-13, -1.72746230e-13, -1.51118782e-13, -1.39004232e-13, -1.23819764e-13,
    -1.02513457e-13, -8.25850415e-14, -6.00154488e-14, -4.85406359e-14, -3.43267861e-14}}};

using LanczosTestF = lanczos_tests<int, float>;
TEST_P(LanczosTestF, Result) { Run(); }

using LanczosTestD = lanczos_tests<int, double>;
TEST_P(LanczosTestD, Result) { Run(); }

using RmatLanczosTestF = rmat_lanczos_tests<int, float>;
TEST_P(RmatLanczosTestF, Result) { Run(); }

using RmatLanczosTestEdgeCase = rmat_lanczos_tests<int, float>;
TEST_P(RmatLanczosTestEdgeCase, Result) { Run(); }

template <typename index_type, typename value_type>
void save_vectors(const std::string& filename,
                  const std::vector<index_type>& rows,
                  const std::vector<index_type>& cols,
                  const std::vector<value_type>& vals)
{
  std::ofstream out(filename, std::ios::binary);

  // Save the size of each vector
  size_t size_rows = rows.size();
  size_t size_cols = cols.size();
  size_t size_vals = vals.size();

  out.write(reinterpret_cast<const char*>(&size_rows), sizeof(size_rows));
  out.write(reinterpret_cast<const char*>(&size_cols), sizeof(size_cols));
  out.write(reinterpret_cast<const char*>(&size_vals), sizeof(size_vals));

  // Save the vectors
  out.write(reinterpret_cast<const char*>(rows.data()), size_rows * sizeof(index_type));
  out.write(reinterpret_cast<const char*>(cols.data()), size_cols * sizeof(index_type));
  out.write(reinterpret_cast<const char*>(vals.data()), size_vals * sizeof(value_type));

  out.close();
}

using DummyLanczosTest = dummy_lanczos_tests<int, float>;
TEST_P(DummyLanczosTest, Result)
{
  raft::resources handle;
  cudaStream_t stream = resource::get_cuda_stream(handle);
  raft::random::RngState rng(42);

  using index_type   = int;
  using value_type   = float;
  int r_scale        = 12;
  int c_scale        = 12;
  float sparsity     = 1;
  uint64_t n_edges   = sparsity * ((long long)(1 << r_scale) * (long long)(1 << c_scale));
  uint64_t n_nodes   = 1 << std::max(r_scale, c_scale);
  uint64_t theta_len = std::max(r_scale, c_scale) * 4;

  std::cout << "n_edges" << n_edges << std::endl;
  std::cout << "n_nodes" << n_nodes << std::endl;

  raft::device_vector<value_type, uint32_t, raft::row_major> theta =
    raft::make_device_vector<value_type, uint32_t, raft::row_major>(handle, theta_len);
  raft::random::uniform<value_type>(handle, rng, theta.view(), 0, 1);
  // print_device_vector("theta", theta.data_handle(), theta_len, std::cout);

  raft::device_matrix<index_type, uint32_t, raft::row_major> out =
    raft::make_device_matrix<index_type, uint32_t, raft::row_major>(handle, n_edges * 2, 2);

  raft::device_vector<index_type, uint32_t, raft::row_major> out_src =
    raft::make_device_vector<index_type, uint32_t, raft::row_major>(handle, n_edges);
  raft::device_vector<index_type, uint32_t, raft::row_major> out_dst =
    raft::make_device_vector<index_type, uint32_t, raft::row_major>(handle, n_edges);

  raft::random::RngState rng1{42};
  raft::random::rmat_rectangular_gen<index_type, value_type>(handle,
                                                             rng1,
                                                             make_const_mdspan(theta.view()),
                                                             out.view(),
                                                             out_src.view(),
                                                             out_dst.view(),
                                                             r_scale,
                                                             c_scale);

  // print_device_vector("out", out.data_handle(), n_edges*2, std::cout);
  // print_device_vector("out_src", out_src.data_handle(), n_edges, std::cout);
  // print_device_vector("out_dst", out_dst.data_handle(), n_edges, std::cout);

  raft::device_vector<value_type, uint32_t, raft::row_major> out_data =
    raft::make_device_vector<value_type, uint32_t, raft::row_major>(handle, n_edges);
  raft::matrix::fill(handle, out_data.view(), 1.0F);
  raft::sparse::COO<value_type, index_type> coo(stream);

  raft::sparse::op::coo_sort(n_nodes,
                             n_nodes,
                             n_edges,
                             out_src.data_handle(),
                             out_dst.data_handle(),
                             out_data.data_handle(),
                             stream);
  raft::sparse::op::max_duplicates<index_type, value_type>(handle,
                                                           coo,
                                                           out_src.data_handle(),
                                                           out_dst.data_handle(),
                                                           out_data.data_handle(),
                                                           n_edges,
                                                           n_nodes,
                                                           n_nodes);

  // print_device_vector("coo_rows", coo.rows(), coo.nnz, std::cout);
  // print_device_vector("coo_cols", coo.cols(), coo.nnz, std::cout);
  // print_device_vector("coo_vals", coo.vals(), coo.nnz, std::cout);

  // print_device_vector("csr_row_indices", row_indices.data_handle(), coo.n_rows + 1, std::cout);

  raft::sparse::COO<value_type, index_type> symmetric_coo(stream);
  raft::sparse::linalg::symmetrize(
    handle, coo.rows(), coo.cols(), coo.vals(), coo.n_rows, coo.n_cols, coo.nnz, symmetric_coo);

  raft::device_vector<index_type, uint32_t, raft::row_major> row_indices =
    raft::make_device_vector<index_type, uint32_t, raft::row_major>(handle,
                                                                    symmetric_coo.n_rows + 1);
  raft::sparse::convert::sorted_coo_to_csr(symmetric_coo.rows(),
                                           symmetric_coo.nnz,
                                           row_indices.data_handle(),
                                           symmetric_coo.n_rows + 1,
                                           stream);

  // print_device_vector("sym_coo_rows", symmetric_coo.rows(), symmetric_coo.nnz, std::cout);
  // print_device_vector("sym_coo_cols", symmetric_coo.cols(), symmetric_coo.nnz, std::cout);
  // print_device_vector("sym_coo_vals", symmetric_coo.vals(), symmetric_coo.nnz, std::cout);

  std::vector<index_type> rowsH(symmetric_coo.n_rows + 1);
  std::vector<index_type> colsH(symmetric_coo.nnz);
  std::vector<value_type> valsH(symmetric_coo.nnz);
  raft::copy(rowsH.data(), row_indices.data_handle(), symmetric_coo.n_rows + 1, stream);
  raft::copy(colsH.data(), symmetric_coo.cols(), symmetric_coo.nnz, stream);
  raft::copy(valsH.data(), symmetric_coo.vals(), symmetric_coo.nnz, stream);

  // This is to inspect the RMAT values and save them to a file
  // save_vectors("sparse.bin", rowsH, colsH, valsH);
}

INSTANTIATE_TEST_CASE_P(LanczosTests, LanczosTestF, ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_CASE_P(LanczosTests, LanczosTestD, ::testing::ValuesIn(inputsd));
INSTANTIATE_TEST_CASE_P(LanczosTests, RmatLanczosTestF, ::testing::ValuesIn(rmat_inputsf));
INSTANTIATE_TEST_CASE_P(LanczosTests,
                        RmatLanczosTestEdgeCase,
                        ::testing::ValuesIn(rmat_inputs_edge_case));

INSTANTIATE_TEST_CASE_P(LanczosTests, DummyLanczosTest, ::testing::ValuesIn(inputsf));

}  // namespace sparse
}  // namespace raft
