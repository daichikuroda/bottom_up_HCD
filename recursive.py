#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""
import numpy as np
import networkx as nx
import scipy.sparse as sp
import sys

# from scipy.sparse.linalg import eigs, eigsh
from scipy.cluster.vq import kmeans2

# import scipy.linalg as splinalg
import scipy.cluster.hierarchy as sch
import utild
import spectrald as spect

# =============================================================================
# TOP-DOWN clustering
# =============================================================================


def non_stop(sG):
    if nx.number_of_nodes(sG) <= 1:
        return False
    else:
        return True


# initial communitites must be numpy array list
def recursive_bipartion(
    G,
    partion_algo=spect.regularized_spectral,
    # stopping_rule=spect.stopping_rule2015,
    stopping_rule=spect.stop_bethe_hessian,
    max_count=False,
    partion_from_big=False,
    non_converge_iter=100,
    initial_communities=None,
):
    if initial_communities is None:
        community_to_divide = [np.array(G.nodes())]
        community_bits_to_go = [[0]]
    else:
        community_to_divide = initial_communities
        community_bits_to_go = [[i] for i in range(len(initial_communities))]
    nodes_per_community = []
    community_bits = []
    count = 0
    while len(community_to_divide) >= 1:
        if partion_from_big:
            li = np.argmax([len(c) for c in community_to_divide])
            nodes = community_to_divide.pop(li)
            cb = community_bits_to_go.pop(li)
        else:
            nodes = community_to_divide.pop()
            cb = community_bits_to_go.pop()
        sG = G.subgraph(nodes)
        # to_continue = utild.roop_for_converge(
        #     stopping_rule, sG, max_roop=non_converge_iter
        # )
        to_continue = stopping_rule(sG)
        if to_continue is True:
            count += 1
            OUT = "Partion count : " + str(count).zfill(2)
            sys.stdout.write("\r%s" % OUT)
            # label, _centroid = utild.roop_for_converge(
            #     partion_algo, sG, max_roop=non_converge_iter, nodelist=nodes
            # )
            label, _centroid = partion_algo(sG, nodelist=nodes)
            nodes_divided = utild.return_communities(label, nodes)
            if len(nodes_divided) >= 2:
                community_to_divide += nodes_divided
                community_bits_to_go += [cb + [0], cb + [1]]
            else:  # in case this fails to split
                print("SPLIT FAILS!!!")
                nodes_per_community.append(nodes)
                community_bits.append(cb)
        elif to_continue is False:
            nodes_per_community.append(nodes)
            community_bits.append(cb)
        else:
            print("to_continue: ", to_continue, type(to_continue))
        if (max_count is not False) and (count >= max_count):
            print("breaked")
            nodes_per_community += community_to_divide
            community_bits += community_bits_to_go
            break
    print("\n")
    return nodes_per_community, community_bits


# =============================================================================
# BOTTOM-UP CLUSTERING
# =============================================================================


def similarity_between(A, label, li, lj):
    iindex = label == li
    jindex = label == lj
    ijindex = np.logical_or(iindex, jindex)
    d = (
        np.sum(A[ijindex, :][:, ijindex])
        - np.sum(A[iindex, :][:, iindex])
        - np.sum(A[jindex, :][:, jindex])
    ) / 2
    size_product = np.sum(label == li) * np.sum(label == lj)
    similarity = d / size_product
    return similarity


def similarity_between2(A, label, li, lj):
    iindex = label == li
    jindex = label == lj
    ijindex = np.logical_or(iindex, jindex)
    d = (
        np.sum(A[ijindex, :][:, ijindex])
        - np.sum(A[iindex, :][:, iindex])
        - np.sum(A[jindex, :][:, jindex])
    ) / 2
    similarity = d
    return similarity


def similarity(G, label, sim_algo=similarity_between):
    A = nx.to_numpy_array(G)
    uq_label = np.unique(label)
    K = len(uq_label)
    similarities = np.zeros((K, K))
    for ik in range(K):
        for jk in range(ik + 1, K):
            li = uq_label[ik]
            lj = uq_label[jk]
            similarity = sim_algo(A, label, li, lj)
            similarities[ik][jk] = similarity
            similarities[jk][ik] = similarity
    return similarities


def condensed_distance_similarity(
    G, label, sim_algo=similarity_between, weighted=False, nodelist=None
):
    if weighted:
        A = nx.to_numpy_array(G, nodelist=nodelist)
    else:
        A = nx.to_numpy_array(G, weight=None, nodelist=nodelist)
    uq_label = np.unique(label)
    K = len(uq_label)
    return np.array(
        [
            sim_algo(A, label, uq_label[ik], uq_label[jk])
            for ik in range(K)
            for jk in range(ik + 1, K)
        ]
    )


def invert(y):
    return 1 / (y + 1)  # add 1 to avoid inf


def subtraction_from_1(y):
    return 1 - y


# it's not completely guranteed that this value does not go below 0
def subtraction_from_max(y, maxy):
    return maxy + 1 - y


def linkage_update_sim_each2(
    similarities2, community_sizes, sim_to_distance=subtraction_from_1
):
    n = int((1 + np.sqrt(len(similarities2) * 8 + 1)) / 2)
    Z = np.zeros((n - 1, 4))
    num_edges_matrix = np.zeros((n, n))  # np.reshape([np.inf] * n ** 2, (n, n))
    ind = 0
    for i in range(n):
        ind1 = ind + (n - i - 1)
        num_edges_matrix[i, np.arange(i + 1, n)] = similarities2[ind:ind1]
        num_edges_matrix[np.arange(i + 1, n), i] = similarities2[ind:ind1]
        ind = ind1
    similarity_matrix = num_edges_matrix / np.outer(community_sizes, community_sizes)
    clusters = list(range(n))
    for i in range(n - 1):
        np.fill_diagonal(similarity_matrix, np.nan)
        argmin = np.nanargmax(similarity_matrix)
        (mini, minj) = (argmin // (n - i), argmin % (n - i))
        if mini > minj:  # deleting from the larger value to avoid misindexing
            ci = clusters.pop(mini)
            cj = clusters.pop(minj)
            comsi = community_sizes.pop(mini)
            comsj = community_sizes.pop(minj)
        elif mini < minj:
            ci = clusters.pop(minj)
            cj = clusters.pop(mini)
            comsi = community_sizes.pop(minj)
            comsj = community_sizes.pop(mini)
        else:
            print("mini == minj", mini, minj)
            raise (ValueError)
        Z[i] = [ci, cj, sim_to_distance(similarity_matrix[mini][minj]), comsi + comsj]
        clusters.append(n + i)
        indexes = np.setdiff1d(np.arange(n - i), [mini, minj])
        new_num_edges = num_edges_matrix[mini] + num_edges_matrix[minj]
        num_edges_matrix = num_edges_matrix[:, indexes][indexes, :]
        num_edges_matrix = np.vstack(
            [
                np.hstack([num_edges_matrix, np.array([new_num_edges[indexes]]).T]),
                [np.hstack([new_num_edges[indexes], [0]])],
            ]
        )
        new_similarities = new_num_edges[indexes] / (
            (comsi + comsj) * np.array(community_sizes)
        )
        community_sizes.append(comsi + comsj)
        similarity_matrix = similarity_matrix[:, indexes][indexes, :]
        similarity_matrix = np.vstack(
            [
                np.hstack([similarity_matrix, np.array([new_similarities]).T]),
                [np.hstack([new_similarities, [0]])],
            ]
        )
    return Z


def bottom_up(
    G,
    bottom_label,
    nodelist=None,
    sim_algo=similarity_between,
    linkage_algo="update_each",
    sim_to_distance=subtraction_from_1,
    weighted=False,
):
    if weighted:
        sim_to_distance = invert
    if linkage_algo == "linkage":
        similarities = condensed_distance_similarity(
            G, bottom_label, sim_algo=sim_algo, weighted=weighted, nodelist=nodelist
        )
        y = sim_to_distance(similarities)
        return sch.linkage(y, method="single")
    elif linkage_algo == "update_each":
        similarities2 = condensed_distance_similarity(
            G,
            bottom_label,
            sim_algo=similarity_between2,
            weighted=weighted,
            nodelist=nodelist,
        )
        community_sizes = [np.sum(bottom_label == l) for l in np.unique(bottom_label)]
        return linkage_update_sim_each2(
            similarities2, community_sizes, sim_to_distance=sim_to_distance
        )
