import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz, load_npz
import time

# Generate a power-law cluster graph to illustrate
import networkx as nx

from scipy.sparse import random
from cupyx.scipy.sparse import random as curandom
import cupy as cp
import scipy



import cupy as cp
import cupy.sparse as csp
import cupyx


def create_uniform_random_csr_matrix(n_rows, n_cols, density, dtype=cp.float32, batches=1):
    """
    Create a uniform random sparse matrix in CSR format using CuPy directly with optimized performance.

    Parameters:
    - n_rows (int): Number of rows in the matrix.
    - n_cols (int): Number of columns in the matrix.
    - density (float): Density of the sparse matrix (fraction of non-zero entries).
    - dtype (cupy.dtype): Data type of the matrix entries (default: cp.float32).

    Returns:
    - CuPy sparse matrix in CSR format.
    """
    if not (0 < density <= 1):
        raise ValueError("Density must be between 0 and 1")

    A = None
    for i in range(batches):
        # Calculate the number of non-zero entries
        num_nonzero = int(n_rows * n_cols * (density/batches))
        print(i, num_nonzero)

        # Generate random row and column indices for non-zero entries
        rows = cp.random.randint(0, n_rows, size=num_nonzero)
        cols = cp.random.randint(0, n_cols, size=num_nonzero)

        # Generate random values for the non-zero entries
        values = cp.random.uniform(low=0.0, high=1.0, size=num_nonzero).astype(dtype)
        X = csp.csr_matrix((values, (rows, cols)), shape=(n_rows, n_cols))
        if A is None:
            A = X
        else:
            A = A + X
    
    return A


def cupy_to_scipy(cupy_csr):
    # Extract data, indices, and indptr from the cupy CSR matrix
    data = cp.asnumpy(cupy_csr.data)
    indices = cp.asnumpy(cupy_csr.indices)
    indptr = cp.asnumpy(cupy_csr.indptr)
    num_nodes = indptr.shape[0] - 1
    
    # Create a scipy CSR matrix using the extracted data
    scipy_csr = csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes), dtype=np.float32)
    
    return scipy_csr


M = 1000000
N = 1000000
density=0.001

while density < 0.1:
    
    # rng = cp.random.RandomState(seed=42)
    # A = curandom(M, N, density=density, format='csr', dtype=cp.float32, random_state=rng)

    try:
        A = create_uniform_random_csr_matrix(M, N, density, dtype=cp.float32, batches=10)
        A = A + A.T

        density=A.nnz / (A.shape[0] * A.shape[1])
        print(A.shape, A.nnz, density)

        A = cupy_to_scipy(A)
        density=A.nnz / (A.shape[0] * A.shape[1])
        print(A.shape, A.nnz, density)

        save_npz(f"uniform_datasets/uniform_nnz{A.nnz}_M{M}.npz", A)
    
    except Exception as e:
        print(e)
        break

    density *= 2

