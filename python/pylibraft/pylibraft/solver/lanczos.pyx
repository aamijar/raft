#
# Copyright (c) 2024-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import cupy as cp
import numpy as np

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uint64_t, uintptr_t

from pylibraft.common import Handle, cai_wrapper, device_ndarray
from pylibraft.common.handle import auto_sync_handle

from libcpp cimport bool

from pylibraft.common.handle cimport device_resources
from pylibraft.random.cpp.rng_state cimport RngState


cdef extern from "raft_runtime/solver/lanczos.hpp" \
        namespace "raft::runtime::solver" nogil:

    cdef void lanczos_solver(
        const device_resources &handle,
        int64_t* rows,
        int64_t* cols,
        double* vals,
        int nnz,
        int n,
        int n_components,
        int max_iterations,
        int ncv,
        double tolerance,
        uint64_t seed,
        double* v0,
        double* eigenvalues,
        double* eigenvectors) except +

    cdef void lanczos_solver(
        const device_resources &handle,
        int64_t* rows,
        int64_t* cols,
        float* vals,
        int nnz,
        int n,
        int n_components,
        int max_iterations,
        int ncv,
        float tolerance,
        uint64_t seed,
        float* v0,
        float* eigenvalues,
        float* eigenvectors) except +

    cdef void lanczos_solver(
        const device_resources &handle,
        int* rows,
        int* cols,
        double* vals,
        int nnz,
        int n,
        int n_components,
        int max_iterations,
        int ncv,
        double tolerance,
        uint64_t seed,
        double* v0,
        double* eigenvalues,
        double* eigenvectors) except +

    cdef void lanczos_solver(
        const device_resources &handle,
        int* rows,
        int* cols,
        float* vals,
        int nnz,
        int n,
        int n_components,
        int max_iterations,
        int ncv,
        float tolerance,
        uint64_t seed,
        float* v0,
        float* eigenvalues,
        float* eigenvectors) except +


@auto_sync_handle
def eigsh(A, k=6, v0=None, ncv=None, maxiter=None,
          tol=0, seed=None, handle=None):
    """
    Find ``k`` eigenvalues and eigenvectors of the real symmetric square
    matrix or complex Hermitian matrix ``A``.

    Solves ``Ax = wx``, the standard eigenvalue problem for ``w`` eigenvalues
    with corresponding eigenvectors ``x``.

    Args:
        a (ndarray, spmatrix or LinearOperator): A symmetric square matrix with
            dimension ``(n, n)``. ``a`` must :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        k (int): The number of eigenvalues and eigenvectors to compute. Must be
            ``1 <= k < n``.
        v0 (ndarray): Starting vector for iteration. If ``None``, a random
            unit vector is used.
        ncv (int): The number of Lanczos vectors generated. Must be
            ``k + 1 < ncv < n``. If ``None``, default value is used.
        maxiter (int): Maximum number of Lanczos update iterations.
            If ``None``, default value is used.
        tol (float): Tolerance for residuals ``||Ax - wx||``. If ``0``, machine
            precision is used.

    Returns:
        tuple:
            It returns ``w`` and ``x``
            where ``w`` is eigenvalues and ``x`` is eigenvectors.

    .. seealso::
        :func:`scipy.sparse.linalg.eigsh`
        :func:`cupyx.scipy.sparse.linalg.eigsh`

    .. note::
        This function uses the thick-restart Lanczos methods
        (https://sdm.lbl.gov/~kewu/ps/trlan.html).

    """

    if A is None:
        raise Exception("'A' cannot be None!")

    rows = A.indptr
    cols = A.indices
    vals = A.data

    rows = cai_wrapper(rows)
    cols = cai_wrapper(cols)
    vals = cai_wrapper(vals)

    IndexType = rows.dtype
    ValueType = vals.dtype

    N = A.shape[0]
    n = N

    rows_ptr = <uintptr_t>rows.data
    cols_ptr = <uintptr_t>cols.data
    vals_ptr = <uintptr_t>vals.data

    if ncv is None:
        ncv = min(n, max(2*k + 1, 20))
    else:
        ncv = min(max(ncv, k + 2), n - 1)

    seed = seed if seed is not None else 42
    if maxiter is None:
        maxiter = 10 * n
    if tol == 0:
        tol = np.finfo(ValueType).eps

    if v0 is None:
        rng = np.random.default_rng(seed)
        v0 = rng.random((N,)).astype(vals.dtype)

    v0 = cai_wrapper(v0)
    v0_ptr = <uintptr_t>v0.data

    eigenvectors = device_ndarray.empty((N, k), dtype=ValueType, order='F')
    eigenvalues = device_ndarray.empty((k), dtype=ValueType, order='F')

    eigenvectors_cai = cai_wrapper(eigenvectors)
    eigenvalues_cai = cai_wrapper(eigenvalues)

    eigenvectors_ptr = <uintptr_t>eigenvectors_cai.data
    eigenvalues_ptr = <uintptr_t>eigenvalues_cai.data

    handle = handle if handle is not None else Handle()
    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    if IndexType == np.int32 and ValueType == np.float32:
        lanczos_solver(
            deref(h),
            <int*> rows_ptr,
            <int*> cols_ptr,
            <float*> vals_ptr,
            <int> A.nnz,
            <int> N,
            <int> k,
            <int> maxiter,
            <int> ncv,
            <float> tol,
            <uint64_t> seed,
            <float*> v0_ptr,
            <float*> eigenvalues_ptr,
            <float*> eigenvectors_ptr,
        )
    elif IndexType == np.int64 and ValueType == np.float32:
        lanczos_solver(
            deref(h),
            <int*> rows_ptr,
            <int*> cols_ptr,
            <float*> vals_ptr,
            <int> A.nnz,
            <int> N,
            <int> k,
            <int> maxiter,
            <int> ncv,
            <float> tol,
            <uint64_t> seed,
            <float*> v0_ptr,
            <float*> eigenvalues_ptr,
            <float*> eigenvectors_ptr,
        )
    elif IndexType == np.int32 and ValueType == np.float64:
        lanczos_solver(
            deref(h),
            <int*> rows_ptr,
            <int*> cols_ptr,
            <float*> vals_ptr,
            <int> A.nnz,
            <int> N,
            <int> k,
            <int> maxiter,
            <int> ncv,
            <float> tol,
            <uint64_t> seed,
            <float*> v0_ptr,
            <float*> eigenvalues_ptr,
            <float*> eigenvectors_ptr,
        )
    elif IndexType == np.int64 and ValueType == np.float64:
        lanczos_solver(
            deref(h),
            <int*> rows_ptr,
            <int*> cols_ptr,
            <float*> vals_ptr,
            <int> A.nnz,
            <int> N,
            <int> k,
            <int> maxiter,
            <int> ncv,
            <float> tol,
            <uint64_t> seed,
            <float*> v0_ptr,
            <float*> eigenvalues_ptr,
            <float*> eigenvectors_ptr,
        )
    else:
        raise ValueError("dtype IndexType=%s and ValueType=%s not supported" %
                         (IndexType, ValueType))

    return (cp.asarray(eigenvalues), cp.asarray(eigenvectors))
