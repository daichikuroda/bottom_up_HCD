#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: daichikuroda
"""

import numpy as np
from skbio import DistanceMatrix
import skbio.tree as sktree
from factor_analyzer.rotator import Rotator
from scipy.linalg import eigh
import itertools as it


def varimax(U, tol=1e-5):
    rotator = Rotator(method="varimax", tol=tol, normalize=False)
    return rotator.fit_transform(U)


# in the original paper sign_adjust is set to False but in T-stochastic paper it is set to True
def vsp(A, k, random_state=0, sign_adjust=True):
    n = A.shape[0]
    eivals, eigvecs = eigh(A)
    indicies = np.argsort(np.absolute(eivals))[::-1][:k]
    U = eigvecs[:, indicies]
    URu = varimax(U)
    hZ = np.sqrt(n) * URu
    if sign_adjust:
        compact_hD = np.sign(np.sum(hZ, axis=0))
        hD = np.diag(compact_hD)
        hZ = hZ @ hD
    return hZ


def tsg_dist(A, hZ, epsilon):
    n = A.shape[0]
    hBnn = hZ.T @ A @ hZ
    hBnn[hBnn < 0] = 0
    hBnn = hBnn / (n**2) + epsilon
    hS_diag = np.diag(hBnn)
    hS = np.diag(np.sqrt(hS_diag))
    hS_inv = np.diag(np.sqrt(hS_diag ** (-1)))
    hB = hS_inv @ hBnn @ hS_inv
    hB[hB > 1] = 1
    hD = -np.log(hB)
    return hD, hS, hB


def synthesis(A, k, epsilon, phi=None):
    import matplotlib.pyplot as plt

    hZ = vsp(A, k)
    hD, hS, hB = tsg_dist(A, hZ, epsilon)
    hD2 = (hD + hD.T) / 2
    np.fill_diagonal(hD2, 0.0)
    hD2 = DistanceMatrix(hD2)
    tree = sktree.nj(hD2, neg_as_zero=False)
    return tree, hZ, hD
