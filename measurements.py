#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools as it

# from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as sch
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import confusion_matrix
from os.path import join
import itertools
import scipy.sparse as sp
import scipy.linalg as splinalg
import utild


def make_cost_m(cm):
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
def calc_accuracy(
    estimated_clustering,
    true_clustering,
    N=None,
    from_clusterings=True,
    different_size=False,
):
    if from_clusterings is True:
        # predicted_labels = utild.communities_to_label_general(estimated_clustering)
        # labels = utild.communities_to_label_general(true_clustering)
        cm = make_confusion_matrix(true_clustering, estimated_clustering)
        if N is None:
            N = sum([len(c) for c in true_clustering])
    else:
        predicted_labels = estimated_clustering
        labels = true_clustering
        cm = confusion_matrix(labels, predicted_labels)
        if N is None:
            N = len(true_clustering)
    indexes = linear_assignment(make_cost_m(cm))
    if different_size:
        cm2 = cm[indexes[0], :][:, indexes[1]]
    else:
        js = [e[1] for e in zip(*sorted(indexes, key=lambda x: x[0]))]
        cm2 = cm[:, js]  # sometimes this causes some errors
    return np.trace(cm2) / N, indexes, cm, cm2


def cb_similarity(xs, ys, minus_one=True):
    s = 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x == y:
            s += 1
        elif x != y:
            break
    if minus_one:
        return s - 1
    else:
        return s


def cb_distance(xs, ys):
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x != y:
            d = len(xs) - i
            break
    else:
        d = 0
    return d


def Sts_P(
    np_A, D, communities, N=None, maxk=None, metric=cb_similarity, arrnage_len=False
):
    def fill_P(P_matrix, cl1, cl2):

        if cl1 == cl2:
            if len(cl1) == 1:
                P_matrix = utild.smart_assigment(
                    P_matrix,
                    cl1,
                    cl1,
                    0,
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
        t0 = max(0, np.shape(D)[0] - len(communities) + 1)
        n = np.shape(D)[0] + 1  # number of clusters
    else:
        t0 = 0
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
        for t in range(t0, n - 1):
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


def calc_cost(np_A, coms):
    S = sum([len(_c) for _c in coms])
    return S * np.sum(
        [
            np.sum(np_A[coms[i], :][:, coms[j]])
            for i in range(len(coms))
            for j in range(i + 1, len(coms))
        ]
    )


def calc_tree_cost(np_A, D, communities, maxk=None):
    n = len(communities)  # number of clusters
    communities = [list(_c) for _c in communities]
    cluster = {i: _c for (i, _c) in zip(range(n), communities)}
    if maxk is None:
        maxk = n - 1
    if len(communities) != n:
        raise ValueError("the number of communities does not match")
    else:
        cost = int(
            np.sum(
                [
                    (
                        len(com)
                        * (
                            np.sum(np_A[:, com][com, :])
                            - np.sum(np.diag(np_A[:, com][com, :]))
                        )
                        / 2
                    )
                    for com in communities
                ]
            )
        )
        print(cost)
        t = 0
        while t <= maxk - 1:
            dist = D[t][2]
            t2 = t
            com_t = {}
            print(t2)
            while D[t2][2] == dist:
                ic0 = int(D[t2][0])
                ic1 = int(D[t2][1])
                comt2 = []
                for ic in (ic0, ic1):
                    if ic in com_t.keys():
                        comt2 += com_t.pop(ic)
                    else:
                        comt2.append(cluster[ic])
                com_t[n + t2] = comt2
                t2 += 1
                if t2 >= maxk:
                    break
            for t3 in range(t, t2):
                cluster[n + t3] = cluster.pop(int(D[t3][0])) + cluster.pop(
                    int(D[t3][1])
                )
            c = calc_cost(np_A, com_t[min(n + t2 - 1, n + maxk)])
            print(min(n + t2 - 1, n + maxk), c)
            cost += c
            t = t2
    return cost


def calc_standrized_square_error_from_matrixes(matrix_est, matrix_true):
    if matrix_est.shape == matrix_true.shape:
        return np.sum((matrix_est - matrix_true) ** 2) / np.sum(matrix_true**2)
    else:
        return np.nan


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


def calc_similarities_for_top_down(A, D, bottom_communities, nodes_list=None):
    n = np.shape(D)[0] + 1  # number of clusters
    if len(bottom_communities) == 1:
        similarities = []
    else:
        cluster = {i: bottom_communities[i] for i in range(n)}
        similarities = []
        for t in range(n - 1):
            c1 = cluster.pop(int(D[t][0]))
            c2 = cluster.pop(int(D[t][1]))
            if nodes_list is None:
                nodes_list = np.array([n for bc in bottom_communities for n in bc])
            mapping = dict(zip(nodes_list, range(len(nodes_list))))
            cc1 = [mapping[n1] for n1 in c1]
            cc2 = [mapping[n2] for n2 in c2]
            cluster[n + t] = np.hstack((c1, c2))
            similarities.append(similarity_between3(A, cc1, cc2))
    return similarities


def calc_old_err_St(matrix_est, matrix_true):
    return calc_standrized_square_error_from_matrixes(matrix_est + 1, matrix_true + 1)


def calc_corref_in_D(
    true_D,
    est_D,
    true_clustering,
    estimated_clustering,
    cm=None,
    N=None,
):
    if N is None:
        N = sum([len(c) for c in true_clustering])
    true_D = np.array(true_D)
    est_D = np.array(est_D)
    if cm is None:
        cm = make_confusion_matrix(true_clustering, estimated_clustering)

    true_v_per_c = np.array([len(c) for c in true_clustering])
    est_v_per_c = np.array([len(c) for c in estimated_clustering])
    true_gamma_outer = np.outer(true_v_per_c, true_v_per_c)
    est_gamma_outer = np.outer(est_v_per_c, est_v_per_c)
    ave_d = np.sum(true_gamma_outer * true_D) / (N**2)
    ave_d_hat = np.sum(est_gamma_outer * est_D) / (N**2)
    ave_d2 = np.sum(true_gamma_outer * true_D**2) / (N**2)
    ave_d_hat2 = np.sum(est_gamma_outer * est_D**2) / (N**2)
    ave_d_d_hat = np.sum(
        [
            np.sum(np.outer(cm[t1, :], cm[t2, :]) * est_D) * true_D[t1, t2]
            for t1, t2 in it.product(range(len(true_clustering)), repeat=2)
        ]
    ) / (N**2)
    corref_in_D = (ave_d_d_hat - ave_d * ave_d_hat) / (
        np.sqrt(ave_d2 - ave_d**2) * np.sqrt(ave_d_hat2 - ave_d_hat**2)
    )
    return corref_in_D


def depth_lcas(community_bits, return_dict=True, ignore_path=False, self_return=False):
    if return_dict and self_return and len(community_bits) >= 1:
        return {
            (bi, bj): cb_similarity(
                community_bits[bi], community_bits[bj], minus_one=False
            )
            for bi, bj in it.combinations_with_replacement(
                range(len(community_bits)), 2
            )
        }
    elif return_dict and self_return and len(community_bits) == 0:
        return {(0, 0): 0}
    elif return_dict:
        return {
            (bi, bj): cb_similarity(
                community_bits[bi], community_bits[bj], minus_one=False
            )
            for bi, bj in it.combinations(range(len(community_bits)), 2)
        }
    else:
        if ignore_path:
            com_bits = np.array(com_bits)
            tree_set = []
            for comb in com_bits.T:
                _, scoms = utild.unique_wt_all_indices_simple(comb)
                tree_set += [tuple(c) for c in scoms]
            tree_set += [tuple(com_bits.T[-1])]
            tree_set = [list(c) for c in set(tree_set)]
            import tree_list_up as tlu

            return tlu.tree_to_matrix(tree_set)
        else:
            return np.array(
                [
                    [
                        cb_similarity(
                            community_bits[bi], community_bits[bj], minus_one=False
                        )
                        for bj in range(len(community_bits))
                    ]
                    for bi in range(len(community_bits))
                ]
            )
