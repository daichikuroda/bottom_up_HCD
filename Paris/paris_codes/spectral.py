import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh
from scipy.cluster.vq import kmeans2
import scipy.cluster.hierarchy as sch
import utild


def spectral(G, k=20):
    # Laplacian matrix
    n = len(G.nodes())
    A = nx.to_scipy_sparse_matrix(G)
    deg = sp.csr_matrix.dot(A, np.ones(n))
    D = sp.diags(deg)
    L = D - A

    # Spectral decomposition
    lam, V = eigsh(L, min(k, n - 1), sigma=-1)
    index = np.argsort(lam)
    lam, V = lam[index], V[:, index]

    return sch.linkage(V, method="ward")
