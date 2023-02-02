#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as sch
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import confusion_matrix
from os.path import join
import itertools
import scipy.sparse as sp
import scipy.linalg as splinalg
import utild


def _make_cost_m(cm):
    s = np.max(cm)
    return -cm + s


def make_confusion_matrix(true_clustering, estimated_clustering):
    return np.array(
        [
            [len(np.intersect1d(est, tre)) for est in estimated_clustering]
            for tre in true_clustering
        ]
    )


# https://smorbieu.gitlab.io/accuracy-from-classification-to-clustering-evaluation/#:~:text=Computing%20accuracy%20for%20clustering%20can,the%20accuracy%20for%20clustering%20results.
# https://en.wikipedia.org/wiki/Hungarian_algorithm
def calc_accuracy(estimated_clustering, true_clustering, N=None, from_clusterings=True):
    if N is None:
        N = sum([len(c) for c in true_clustering])
    if from_clusterings is True:
        # predicted_labels = utild.communities_to_label_general(estimated_clustering)
        # labels = utild.communities_to_label_general(true_clustering)
        cm = make_confusion_matrix(true_clustering, estimated_clustering)
    else:
        predicted_labels = estimated_clustering
        labels = true_clustering
        cm = confusion_matrix(labels, predicted_labels)
    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in zip(*sorted(indexes, key=lambda x: x[0]))]
    cm2 = cm[:, js]
    return np.trace(cm2) / N, indexes, cm, cm2


def cb_similarity(xs, ys):
    s = 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x == y:
            s += 1
        elif x != y:
            break
    return s - 1


def cb_distance(xs, ys):
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x != y:
            d = len(xs) - i
            break
    else:
        d = 0
    return d


def estimate_p_matrix_simple(np_A, clustering):
    B = np.zeros((len(clustering), len(clustering)))
    for icl1 in range(len(clustering)):
        cl1 = clustering[icl1]
        for icl2 in range(icl1, len(clustering)):
            cl2 = clustering[icl2]
            p = np.sum(np_A[cl1, :][:, cl2]) / (len(cl1) * len(cl2))
            B[icl1][icl2] = p
            B[icl2][icl1] = p
            B[icl1][icl1] = np.sum(np_A[cl1, :][:, cl1]) / (len(cl1 ** 2))
            B[icl2][icl2] = np.sum(np_A[cl2, :][:, cl2]) / (len(cl2 ** 2))
    return B


def estimate_p_matrix(np_A, D, communities, maxk=None):
    if maxk is None:
        maxk = len(communities)
    P = np.zeros(np_A.shape)
    clustering = utild.clustering_k_communities(D, maxk, communities)
    for icl1 in range(len(clustering)):
        cl1 = clustering[icl1]
        if len(cl1) == 1:
            P = utild.smart_assigment(P, cl1, cl1, 0,)
        else:
            P = utild.smart_assigment(
                P, cl1, cl1, np.sum(np_A[cl1, :][:, cl1]) / (len(cl1) * (len(cl1) - 1))
            )
        # P[cl1, :][:, cl1] = np.sum(np_A[cl1, :][:, cl1]) / (len(cl1) ** 2)
        for icl2 in range(icl1 + 1, len(clustering)):
            cl2 = clustering[icl2]
            p = np.sum(np_A[cl1, :][:, cl2]) / (len(cl1) * len(cl2))
            # P[cl1, :][:, cl2] = p
            P = utild.smart_assigment(P, cl1, cl2, p)
            # P[cl2, :][:, cl1] = p
            P = utild.smart_assigment(P, cl2, cl1, p)
    for k in range(2, maxk)[::-1]:
        clustering = utild.clustering_k_communities(D, k, communities)
        for icl1 in range(len(clustering)):
            cl1 = clustering[icl1]
            for icl2 in range(icl1 + 1, len(clustering)):
                cl2 = clustering[icl2]
                p = np.sum(np_A[cl1, :][:, cl2]) / (len(cl1) * len(cl2))
                # P[cl1, :][:, cl2] = p
                P = utild.smart_assigment(P, cl1, cl2, p)
                # P[cl2, :][:, cl1] = p
                P = utild.smart_assigment(P, cl2, cl1, p)
    return P


def tree_distance_matrix(
    D, communities, N=None, maxk=None, metric=cb_distance, arrnage_len=True
):
    if N is None:
        N = sum([len(c) for c in communities])
    if maxk is not None:
        communities, cluster = utild.clustering_k_communities(
            D, maxk, communities, return_cluster=True
        )
        t = max(0, len(D) - len(communities) + 1)
        n = np.shape(D)[0] + 1  # number of clusters
    else:
        t = 0
        n = np.shape(D)[0] + 1  # number of clusters
        cluster = {i: [i] for i in range(n)}
    cluster_bits = {i: [i] for i in range(n)}
    cluster_bits = {k: [i] for i, k in enumerate(cluster.keys())}
    cluster_new = {k: [k] for k in cluster.keys()}
    if len(communities) <= 1:
        St_small = np.zeros((1, 1), dtype=int)
    else:
        for t in range(t, n - 1):
            c0 = int(D[t][0])
            c1 = int(D[t][1])
            cluster[n + t] = cluster.pop(c0) + cluster.pop(c1)
            cb0 = cluster_new.pop(c0)
            cb1 = cluster_new.pop(c1)
            cluster_new[n + t] = cb0 + cb1
            for c in cb0 + cb1:
                cluster_bits[c].append(n + t)
        cluster_bits = [cb[::-1] for cb in cluster_bits.values()]
        if arrnage_len:
            cluster_bits = utild.arrange_len_community_bits(cluster_bits)
        St_small = np.array(
            [[metric(cb0, cb1) for cb1 in cluster_bits] for cb0 in cluster_bits],
            dtype=int,
        )
    St = utild.st_small_to_st(St_small, communities)
    return St, St_small


def Sts_P(
    np_A, D, communities, N=None, maxk=None, metric=cb_similarity, arrnage_len=False
):
    def fill_P(P_matrix, cl1, cl2):

        if cl1 == cl2:
            if len(cl1) == 1:
                P_matrix = utild.smart_assigment(
                    P_matrix, cl1, cl1, 0,
                )  # num edges is np_A[cl1, :][:, cl1]/2
            else:
                P_matrix = utild.smart_assigment(
                    P_matrix,
                    cl1,
                    cl1,
                    np.sum(np_A[cl1, :][:, cl1]) / (len(cl1) * (len(cl1) - 1)),
                )  # num edges is np_A[cl1, :][:, cl1]/2
        else:
            p = np.sum(np_A[cl1, :][:, cl2]) / (
                len(cl1) * len(cl2)
            )  # num edges is np_A[cl1, :][:, cl2] or (np_A[cl1, :][:, cl2]+np_A[cl2, :][:, cl1])/2
            P_matrix = utild.smart_assigment(P_matrix, cl1, cl2, p)
            P_matrix = utild.smart_assigment(P_matrix, cl2, cl1, p)
        return P_matrix

    if N is None:
        # N = sum([len(c) for c in communities])
        N = len(np_A)
    if maxk is not None:
        communities, cluster, cluster_com = utild.clustering_k_communities(
            D,
            maxk,
            communities,
            return_cluster=True,
            return_cluster_community_dict=True,
        )
        t = max(0, np.shape(D)[0] - len(communities) + 1)
        n = np.shape(D)[0] + 1  # number of clusters
    else:
        t = 0
        n = np.shape(D)[0] + 1  # number of clusters
        cluster = {i: [i] for i in range(n)}
        cluster_com = {i: list(c) for i, c in enumerate(communities)}
    cluster_new = {k: [k] for k in cluster.keys()}
    cluster_bits = {k: [i] for i, k in enumerate(cluster.keys())}

    if len(communities) <= 1:
        St_small = np.zeros((1, 1), dtype=int)
        P_matrix = np.ones(np_A.shape) * (
            np.sum(np_A) / (len(np_A) * (len(np_A) - 1))
        )  # num edges is {sum(np_A)/2} / {N*(N-1)/2}
    else:
        P_matrix = np.zeros(np_A.shape)
        for _c in cluster_com.values():
            P_matrix = fill_P(P_matrix, _c, _c)
        for t in range(t, n - 1):
            c0 = int(D[t][0])
            c1 = int(D[t][1])
            oc0 = cluster.pop(c0)
            oc1 = cluster.pop(c1)
            cc0 = cluster_com.pop(c0)
            cc1 = cluster_com.pop(c1)
            cb0 = cluster_new.pop(c0)
            cb1 = cluster_new.pop(c1)
            cluster[n + t] = oc0 + oc1
            cluster_com[n + t] = cc0 + cc1
            cluster_new[n + t] = cb0 + cb1
            for c in cb0 + cb1:
                cluster_bits[c].append(n + t)

            P_matrix = fill_P(P_matrix, cc0, cc1)
        cluster_bits = [cb[::-1] for cb in cluster_bits.values()]
        if arrnage_len:
            cluster_bits = utild.arrange_len_community_bits(cluster_bits)
        St_small = np.array(
            [[metric(cb0, cb1) for cb1 in cluster_bits] for cb0 in cluster_bits],
            dtype=int,
        )
    St = utild.st_small_to_st(St_small, communities)
    return St, St_small, P_matrix


# to output the similarity of the "top-down" paper, you need to add 1 to the matirx
def tree_similarity_matrix(D, communities, N=None, maxk=None):
    return tree_distance_matrix(
        D, communities, N=N, maxk=maxk, metric=cb_similarity, arrnage_len=False
    )


def calc_standrized_square_error_from_matrixes(matrix_est, matrix_true):
    if matrix_est.shape == matrix_true.shape:
        return np.sum((matrix_est - matrix_true) ** 2) / np.sum(matrix_true ** 2)
    else:
        return np.nan


def calc_err_est_p_matrix(np_A, D, communities, P_true, maxk=None):
    P_est = estimate_p_matrix(np_A, D, communities, maxk)
    return np.sum((P_est - P_true) ** 2) / np.sum(P_true ** 2)  # , P_est


def similarity_between3(A, community1, community2):
    commerge = np.hstack((community1, community2))
    d = (
        np.sum(A[commerge, :][:, commerge])
        - np.sum(A[community1, :][:, community1])
        - np.sum(A[community2, :][:, community2])
    ) / 2
    size_product = len(community1) * len(community2)
    similarity = d / size_product
    return similarity


def calc_similarities_for_top_down(A, D, communities, nodes_map=None):
    n = np.shape(D)[0] + 1  # number of clusters
    cluster = {i: communities[i] for i in range(n)}
    similarities = []
    for t in range(n - 1):
        c1 = cluster.pop(int(D[t][0]))
        c2 = cluster.pop(int(D[t][1]))
        if nodes_map is not None:
            cc1 = [nodes_map[n1] for n1 in c1]
            cc2 = [nodes_map[n2] for n2 in c2]
        cluster[n + t] = np.hstack((c1, c2))
        similarities.append(similarity_between3(A, cc1, cc2))
    # cluster_community = [
    #     select_several_indices_in_list(communities, c) for c in cluster.values()
    # ]
    return similarities


def calc_old_err_St(matrix_est, matrix_true):
    return calc_standrized_square_error_from_matrixes(matrix_est + 1, matrix_true + 1)
