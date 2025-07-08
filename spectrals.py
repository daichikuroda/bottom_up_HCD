#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh
from scipy.cluster.vq import kmeans2
import scipy.linalg as splinalg
import scipy.cluster.hierarchy as sch
import utils
import measurements as mea


def return_Lrw(W):
    d_inverse = np.array(W.sum(axis=0, dtype=float))[0] ** (-1)
    d_inverse[np.isinf(d_inverse) + np.isnan(d_inverse)] = 0  # this is to deal inf
    Lrw = sp.identity(W.shape[0]) - W.multiply(
        np.reshape(d_inverse, (len(d_inverse), 1))
    )
    return Lrw


def return_Lsym(W, d):
    d_12 = d ** (-1 / 2)
    d_12[np.isinf(d_12) + np.isnan(d_12)] = 0  # this is to deal inf
    D_12 = sp.identity(W.shape[0]).multiply(
        d_12
    )  # may be able to be faster than this if you implemnt this like def return_Lrw(W)
    return sp.identity(W.shape[0]) - D_12 * W * D_12


def regularized_spectral(G, tau=0.1, k=2, nodelist=None):
    # regularized Laplacian matrix
    n = len(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodelist)
    d = np.array(A.sum(axis=0))
    Atau = A + tau * np.mean(d) / n
    dtau = np.array(Atau.sum(axis=0))
    Ltau = return_Lsym(Atau, dtau)
    lam, V = splinalg.eigh(Ltau)
    index = np.argsort(lam)[: min(k, n - 1)]
    lam, V = lam[index], V[:, index]
    centroid, label = kmeans2(V, k, minit="++")
    return label, centroid


# stopping rule: non-backtracking method of Le ad Levina (2015)
def stopping_rule2015(G, k=2):
    n = len(G.nodes())
    if n <= 2:
        to_continue = False
    else:
        A = nx.to_numpy_array(G)
        d = np.array(A.sum(axis=0, dtype=float))
        Bnb = sp.vstack(
            [
                sp.hstack([0 * sp.identity(n), sp.diags(d, 0) - sp.identity(n)]),
                sp.hstack([-sp.identity(n), A]),
            ]
        )
        vals = eigs(Bnb, min(k, n - 1), which="LR", return_eigenvectors=False)
        vals2 = eigs(Bnb, min(k, n - 1), which="SR", return_eigenvectors=False)
        vals = np.unique(list(vals) + list(vals2))
        if len(vals) < k:
            to_continue = False
        else:
            real_vals = abs(vals.real)
            # kth eignevalue is indexed as k - 1
            to_continue = real_vals[np.argsort(real_vals)[::-1]][k - 1] >= (
                np.sum(d**2) / np.sum(d) - 1
            ) ** (1 / 2)
    return to_continue


# for the dendrogram
def linkage_for_recursive_algo(community_bits):
    if len(community_bits) == 1:
        return np.array([[0, 0, 0, 0]])
    else:
        community_bits = utils.arrange_len_community_bits(community_bits)
        return sch.linkage(community_bits, method="single", metric=mea.cb_distance)


def betheHessian(G, r=None, weighted=False, nodelist=None):
    n = len(G.nodes())
    if weighted:
        A = nx.adjacency_matrix(G, nodelist=nodelist)
    else:
        A = nx.adjacency_matrix(G, nodelist=nodelist, weight=None)
    degrees = A.sum(axis=0, dtype=float)
    if r == None:
        d = np.array(degrees)
        r = np.sum(d**2) / float(
            np.sum(d) - 1
        )  # it somehow omits nan when calculating 0/(-1)
        r = np.sqrt(r)

    if weighted:
        mapping = dict(zip(G, range(0, n)))
        GG = nx.relabel_nodes(G, mapping)
        A = A.todense()
        H = np.zeros((n, n))
        for i in range(n):
            neigh_i = [u for u in GG.neighbors(i)]
            H[i, i] = 1 + np.sum(
                [A[i, j] ** 2 / (r**2 - A[i, j] ** 1) for j in neigh_i]
            )
            for j in neigh_i:
                H[i, j] = -r * A[i, j] / (r**2 - A[i, j])
        H = (r**2 - 1) * sp.csr_matrix(H)
    else:
        D = sp.identity(A.shape[0]).multiply(degrees)
        H = (r**2 - 1) * sp.eye(n) + D - r * A
    return H


def stop_bethe_hessian(G, k=2, r=None, weighted=False):
    if len(G.nodes()) <= 1:
        to_continue = False
    else:
        lam, V = bethe_hessian_spectrum(G, r=None, weighted=False)
        index = lam < 0
        to_continue = np.count_nonzero(index) >= k
    return to_continue


def bethe_hessian_spectrum(G, r=None, weighted=False):
    return splinalg.eigh(betheHessian(G, r, weighted=weighted).todense())
