import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz, load_npz
import time

# Generate a power-law cluster graph to illustrate
import networkx as nx


M = 100000
N = 50

for i in range(5):
    density=0

    prevN = N
    while density < 0.5:

        try:
            start = time.time()
            G = nx.powerlaw_cluster_graph(M, N, 0.1)  # Example graph
            degrees = np.array([d for n, d in G.degree()])

            print("hi")

            A = nx.to_scipy_sparse_array(G)
            density=A.nnz / (A.shape[0] * A.shape[1])
            print(A.shape, A.nnz, density)

            save_npz(f'datasets/powerlaw_nnz{A.nnz}_M{M}_N{N}_.npz', A)
            print(f'datasets/powerlaw_nnz{A.nnz}_M{M}_N{N}_.npz', time.time() - start)
        
        except Exception as e:
            print(e)
            break

        N *= 2
    N = prevN
    M *= 10
    N *= 10



