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
from os.path import join
import itertools
import scipy.sparse as sp
import scipy.linalg as splinalg


line_styles = ["-", "--", "-.", ":", (0, (1, 10))]
markers = [
    "o",
    "v",
    "s",
    "P",
    "x",
    ".",
    ",",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "p",
    "*",
    "h",
    "H",
    "+",
    "X",
    "D",
    "d",
    "|",
    "_",
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
]
colors = [
    "b",
    "g",
    "r",
    "c",
    "m",
    "y",
    "k",
    "0.5",
    "0.3",
    "0.8",
    "0.6",
    "0.2",
    "0.7",
    "0.1",
    "0.9",
] + list(plt.cm.tab20c((4.0 / 3 * np.arange(20 * 3 / 4)).astype(int)))


def run_time_algo(G, algo):
    t0 = time.time()
    C = algo(G)
    t1 = time.time()
    print(t1 - t0)
    return C


def run_time_algo2(algo, *params, **options):
    t0 = time.time()
    C = algo(*params, **options)
    t1 = time.time()
    t = t1 - t0
    print(t)
    return C, t


def calc_p_from_a(a, N):
    return np.array(a) * np.log(N) / N


def calc_overlapped_portion(c1, c2):
    return len(set(c1) & set(c2)) / len(set(c1) | set(c2))


def smart_indexing(llist, indexes):
    return [llist[i] for i in indexes]


def smart_indexing_by_bools(llist, bools):
    return [llist[i] for (i, b) in enumerate(bools) if b]


def smart_assigment(np_array, indexes1, indexes2, assign_value):
    a = np_array[indexes1, :]
    a[:, indexes2] = assign_value
    np_array[indexes1, :] = a
    return np_array  # no need to return this but for the clarification


def roop_for_converge(func, *params, max_roop=10):
    for i in range(max_roop - 1):
        try:
            r = func(*params)
        except sp.linalg.ArpackNoConvergence as e:
            print(e)
            pass
        except splinalg.LinAlgError as e:
            print(e)
            pass
        else:
            break
    else:
        r = func(*params)
    return r


def return_community(label, np_nodes, ci=0):
    return np_nodes[label == ci]


def return_communities(label, np_nodes=None, return_as_dict=False):
    if np_nodes is None:
        np_nodes = np.arange(len(label))
    if return_as_dict:
        communities = {}
        for ci in np.unique(label):
            communities[ci] = return_community(label, np_nodes, ci)
    else:
        communities = [return_community(label, np_nodes, ci) for ci in np.unique(label)]
    return communities


def communities_to_label(communities, nodes_list=None):
    label = np.zeros(sum([len(c) for c in communities]), dtype=int)
    if nodes_list is None:
        for l, c in enumerate(communities):
            label[c] = l
    elif len(nodes_list) == len(label):
        _mapping = dict(zip(nodes_list, range(len(label))))
        for l, c in enumerate(communities):
            label[[_mapping[n] for n in c]] = l
    return label


# the label is in the ascending order of the nodes numbers
def communities_to_label_general(communities):
    communities = [list(c) for c in communities]
    mapping = dict(
        zip(
            sorted(sum(communities, [])),
            np.arange(sum([len(c) for c in communities])),
        )
    )
    label = np.zeros(sum([len(c) for c in communities]), dtype=int)
    for l, c in enumerate(communities):
        for cc in c:
            label[mapping[cc]] = l
    return label  # , mapping


def arrange_len_community_bits(community_bits):
    maxl = max([len(cb) for cb in community_bits])
    return [cb + [0] * (maxl - len(cb)) for cb in community_bits]


def st_small_to_st(St_small, communities, N=None):
    if N is None:
        N = sum([len(c) for c in communities])
    St = np.zeros((N, N), dtype=int)
    for icl0, cl0 in enumerate(communities):
        St = smart_assigment(St, cl0, cl0, St_small[icl0][icl0])
        for icl1 in range(icl0 + 1, len(communities)):
            cl1 = communities[icl1]
            St = smart_assigment(St, cl0, cl1, St_small[icl0][icl1])
            St = smart_assigment(St, cl1, cl0, St_small[icl0][icl1])

    return St


def select_several_indices_in_list(llist, indices):
    return sum([list(llist[i]) for i in indices], [])


# return consensed_distance_matrix for scipy linkage parameter y
# upper triangular of distance matrix
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
def condensed_distance_matrix(distance_matrix):
    return sum([list(d[(i + 1) :]) for (i, d) in enumerate(distance_matrix)], [])


def distance_matrix_to_condensed_one(distance_matrix):
    K = len(distance_matrix)
    return np.array(
        [distance_matrix[ik][jk] for ik in range(K) for jk in range(ik + 1, K)]
    )


def nx_dendrogram(D, communities, from_community_bits=False):
    n = len(D) + 1  # number of clusters
    if not from_community_bits and len(communities) != n:
        raise ValueError("the number of communities does not match")
    elif from_community_bits and len(communities) == n - 1:
        n = n - 1
    elif from_community_bits and len(communities) != n - 1:
        raise ValueError("the number of communities does not match")
    cluster = {i: list(communities[i]) for i in range(n)}
    dG = nx.DiGraph()
    nodes = sum([list(c) for c in communities], [])
    # dG.add_nodes_from(nodes)
    num_for_tree = 10 ** len(str(max(nodes)))
    mapping_encode = dict(zip(nodes, range(len(nodes))))
    mapping_decode = dict(zip(range(len(nodes)), nodes))

    for ic, _c in enumerate(communities):
        if len(_c) >= 2:
            for _node in _c:
                dG.add_edge(num_for_tree + ic, mapping_encode[_node])
        elif len(_c) == 1:
            _node = _c[0]
            dG.add_node(
                num_for_tree + ic,
            )
            mapping_decode[num_for_tree + ic] = _node
        else:
            print("there is empty community")
        # print(num_for_tree + ic)
    if len(cluster) == 1:
        cluster_community = communities
    else:
        if from_community_bits:
            community_bits = D
            community_bits = np.array(arrange_len_community_bits(community_bits))
            tree_height = np.shape(community_bits)[1]
            t = num_for_tree
            community_bits2 = community_bits.copy()
            for h in range(1, tree_height):
                ucbs = np.unique(np.array(community_bits2[:, :-1]), axis=0)
                mcoms = {
                    (imc + t + len(community_bits2)): [
                        cn + t
                        for cn, cb in enumerate(community_bits2[:, :-1])
                        if np.all(cb == ucb)
                    ]
                    for imc, ucb in enumerate(ucbs)
                }
                for imc, coms in mcoms.items():
                    print("add node", imc)
                    print("add edge", coms)
                    dG.add_node(imc, distance=h)
                    for com in coms:
                        dG.add_edge(imc, com)
                t += len(community_bits2)
                community_bits2 = ucbs

        else:
            for t in range(n - 1):
                # print(t, cluster)
                ic0 = int(D[t][0])
                ic1 = int(D[t][1])
                c0 = cluster.pop(ic0)
                c1 = cluster.pop(ic1)
                dG.add_node(n + t + num_for_tree, distance=D[t][2])
                dG.add_edge(n + t + num_for_tree, ic0 + num_for_tree)
                dG.add_edge(n + t + num_for_tree, ic1 + num_for_tree)
                # print(n + t + num_for_tree)
                cluster[n + t] = c0 + c1

    # cluster_community = [
    #     select_several_indices_in_list(communities, c) for c in cluster.values()
    # ]
    dG = nx.relabel_nodes(dG, mapping_decode)
    return dG, list(cluster.values())


def dG_with_distance(dG, logscale=True, resolution=100, last_size=1/30):
    def clog(x):
        if x is not None:
            return np.log(x)

    def none_convert(x):
        return x

    if logscale:
        convert = clog
    else:
        convert = none_convert
    dG2 = nx.DiGraph()
    dG.nodes("distance")
    node_distance_dict = dict(dG.nodes("distance"))
    start = max(dG.nodes())
    _sd = convert(node_distance_dict[start])
    dG2.add_node(start, distance=_sd)
    currents = [start]
    unit = _sd / resolution
    ed = (
        convert(min([_n for _n in node_distance_dict.values() if _n is not None]))
        - _sd * last_size
    )
    bnn = 0
    while len(currents) >= 1:
        _c = currents.pop()
        _cd0 = node_distance_dict[_c]
        _cd = convert(_cd0)
        if _cd is None:
            _cd = ed
        _nexts = list(dG.successors(_c))
        # print(_nexts)
        currents += _nexts
        for _n in _nexts:
            _nd = convert(node_distance_dict[_n])
            if _nd is None and _cd0 is None:
                _num_nodes_between = int(resolution * last_size)
            elif _nd is None:
                _nd = ed
                # print(_c, _n, _cd, _nd)
                _num_nodes_between = int((_cd - _nd) // unit)
            else:
                _num_nodes_between = int((_cd - _nd) // unit)
            dG2.add_node(_n, distance=_nd)
            if _num_nodes_between <= 1:
                dG2.add_edge(_c, _n)
            elif _num_nodes_between >= 2:
                bnn -= 1
                dG2.add_edge(_c, bnn)
                for _nb in range(_num_nodes_between - 1):
                    # print(bnn)
                    bnn -= 1
                    dG2.add_edge(bnn + 1, bnn)
                dG2.add_edge(bnn, _n)
    return dG2


def theoritical_threashold(K, N):
    return ((K * np.log(N)) / N) ** (1 / 2)


def create_group_matrix(num_layer, num_child=2):
    group_matrix = np.ones(
        (num_child ** (num_layer - 1), num_child ** (num_layer - 1)), dtype=int
    ) * (num_layer - 1)
    for l in range(num_layer):
        num_groups = num_child**l
        num_in_group = num_child ** (num_layer - l - 1)
        for ig in range(num_groups):
            group_matrix[
                ig * num_in_group : (ig + 1) * num_in_group,
                ig * num_in_group : (ig + 1) * num_in_group,
            ] = (
                l
            )
    return group_matrix


# Select the clustering after k merges
def clustering_k_merge(
    D, k, communities, return_cluster=False, return_cluster_community_dict=False
):
    n = np.shape(D)[0] + 1  # number of clusters
    k = min(k, n - 1)
    cluster = {i: [i] for i in range(n)}
    if len(communities) == 1:
        cluster_community = communities
        cluster_community_dict = {0: communities}
    elif len(communities) != n:
        raise ValueError("the number of communities does not match")
    else:
        for t in range(k):
            cluster[n + t] = cluster.pop(int(D[t][0])) + cluster.pop(int(D[t][1]))
        cluster_community = [
            select_several_indices_in_list(communities, c) for c in cluster.values()
        ]
        cluster_community_dict = {
            k: select_several_indices_in_list(communities, c)
            for k, c in cluster.items()
        }
    if return_cluster_community_dict and return_cluster:
        return cluster_community, cluster, cluster_community_dict
    elif return_cluster_community_dict:
        return cluster_community, cluster_community_dict
    elif return_cluster:
        return cluster_community, cluster
    else:
        return cluster_community


def clustering_k_communities(
    D, k, communities, return_cluster=False, return_cluster_community_dict=False
):
    k = len(communities) - k
    return clustering_k_merge(
        D,
        k,
        communities,
        return_cluster=return_cluster,
        return_cluster_community_dict=return_cluster_community_dict,
    )


def simple_ave(X, X0):
    return np.mean(abs(X - X0))


def l21(X, X0):
    return np.mean(np.sqrt(np.sum((X - X0) ** 2, axis=0)))
