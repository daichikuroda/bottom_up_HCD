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


def lca_to_distance_matrix(lca_matrix):
    lca_matrix = np.array(lca_matrix)
    # Calculate the depth of each leaf node from the root
    depths = np.max(lca_matrix, axis=0)
    # Calculate the distance between each pair of nodes
    tree_distance_matrix = np.add.outer(depths, depths) - 2 * lca_matrix

    return tree_distance_matrix


def calc_p_from_a(a, N):
    return np.array(a) * np.log(N) / N


def smart_assigment(np_array, indexes1, indexes2, assign_value):
    a = np_array[indexes1, :]
    a[:, indexes2] = assign_value
    np_array[indexes1, :] = a
    return np_array  # no need to return this but for the clarification


def select_several_indices_in_list(llist, indices):
    return sum([list(llist[i]) for i in indices], [])


def roop_for_converge(func, *params, **options):
    max_roop = options.pop("max_roop", 10)
    for i in range(max_roop - 1):
        try:
            r = func(*params, **options)
        except sp.linalg.ArpackNoConvergence as e:
            print(e)
            pass
        except splinalg.LinAlgError as e:
            print(e)
            pass
        else:
            break
    else:
        r = func(*params, **options)
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
                    dG.add_node(imc, distance=h)
                    for com in coms:
                        dG.add_edge(imc, com)
                t += len(community_bits2)
                community_bits2 = ucbs

        else:
            for t in range(n - 1):
                ic0 = int(D[t][0])
                ic1 = int(D[t][1])
                c0 = cluster.pop(ic0)
                c1 = cluster.pop(ic1)
                dG.add_node(n + t + num_for_tree, distance=D[t][2])
                dG.add_edge(n + t + num_for_tree, ic0 + num_for_tree)
                dG.add_edge(n + t + num_for_tree, ic1 + num_for_tree)
                cluster[n + t] = c0 + c1

    dG = nx.relabel_nodes(dG, mapping_decode)
    return dG, list(cluster.values())


def dG_with_distance(dG, logscale=True, resolution=100, last_size=1 / 30):
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
            ] = l
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


def clustering_k_communities_by_similarities(D, k, communities, similarities):
    # n = np.shape(D)[0] + 1  # number of clusters
    n = len(communities)
    cluster = {i: communities[i] for i in range(n)}
    sim_argsorted = list(np.argsort(similarities))
    i = 1
    while len(cluster) > k:
        t = sim_argsorted[-i]
        if int(D[t][0]) in cluster.keys() and int(D[t][1]) in cluster.keys():
            t = sim_argsorted.pop(-i)
            cluster[n + t] = np.hstack(
                (cluster.pop(int(D[t][0])), cluster.pop(int(D[t][1])))
            )
            i = 1
        else:
            i += 1

    return list(cluster.values())


def simple_ave(X, X0):
    return np.mean(abs(X - X0))


def l21(X, X0):
    return np.mean(np.sqrt(np.sum((X - X0) ** 2, axis=0)))


def drop_edges(G, q, excepts=None, parent_seed=None):
    G2 = G.copy()
    _rng = np.random.default_rng(parent_seed)
    edges = np.array(list(set(G2.edges()) - set(excepts)))
    _edges_to_rm = edges[_rng.random(len(edges)) < q]
    G2.remove_edges_from(_edges_to_rm)
    return G2


def add_noise_edges(G, q, only=None, excepts=None, parent_seed=None):
    G2 = G.copy()
    _rng = np.random.default_rng(parent_seed)
    if only is not None:
        edges = only
    else:
        edges = itertools.combinations(G.nodes(), 2)
    # edges_candidates = np.array(list(set(edges) - set(excepts)))
    if excepts is None:
        excepts = []
    _edges_to_add = []
    for _e in edges:
        if (
            _rng.random() < q
            and (_e not in excepts)
            and ((_e[1], _e[0]) not in excepts)
        ):
            _edges_to_add.append(_e)
    G2.add_edges_from(_edges_to_add)
    return G2


def network_from_edgelist(edge_list, weighted=True, add_nodes=None):
    G = nx.Graph()
    if add_nodes is not None:
        G.add_nodes_from(add_nodes)
    for i, j, w in edge_list:
        if weighted:
            G.add_edge(int(i), int(j), weight=w)
        else:
            G.add_edge(int(i), int(j))
    return G


def fluctuate_edge_weights(G, sigma, seed=None, weight_convert="log"):
    _rng = np.random.default_rng(seed)
    wes = G.edges(data="weight")
    fluctions = sigma * _rng.standard_normal(len(wes))
    if weight_convert == "log":
        wes_new = [
            (e0, e1, np.log(w + f))
            for ((e0, e1, w), f) in zip(wes, fluctions)
            if w + f > 1
        ]
    elif weight_convert == "log+1":
        wes_new = [
            (e0, e1, np.log(w + f + 1))
            for ((e0, e1, w), f) in zip(wes, fluctions)
            if w + f > 0
        ]
    else:
        wes_new = [
            (e0, e1, w + f) for ((e0, e1, w), f) in zip(wes, fluctions) if w + f > 0
        ]
    G2 = network_from_edgelist(wes_new, add_nodes=sorted(G.nodes))
    return G2


def return_supercoms(Z, bottom_communities):
    num_bottom = len(bottom_communities)
    supercoms = {
        (iz + num_bottom): [
            [int(_i) for _i in z[:2]],
            [int(_i) for _i in z[:2]],
            [iz + num_bottom],
            z[2],
        ]
        for iz, z in enumerate(Z)
    }
    potential_merges = []
    for iz, z in supercoms.items():
        zaa = []
        for c in z[0]:
            if c >= num_bottom:
                zaa += supercoms[c][1]
                potential_merges.append((iz, c))
            else:
                zaa.append(c)
        supercoms[iz][1] = zaa
    return supercoms, potential_merges


def supercoms_to_community_bits(supercoms):
    if len(supercoms) == 0:
        return []
    cbs = [[b] for b in np.sort(supercoms[max(supercoms.keys())][1])]
    num_b = len(cbs)
    for js, s in enumerate(list(supercoms.values())[:-1]):  # the last one is the root
        for ss in s[1]:
            cbs[ss].append(js + num_b)
    cbs = [cb[::-1] for cb in cbs]
    return cbs
