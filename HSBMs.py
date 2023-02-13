#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""

import numpy as np
import networkx as nx
import itertools

import utild
import measurements as mea
import positions


def create_perfect_tree(depth, num_children):
    G = nx.Graph()
    nn = 1
    parents = [0]
    for d in range(depth):
        new_parents = []
        for parent in parents:
            for child in range(num_children):
                G.add_edge(parent, nn)
                new_parents.append(nn)
                nn += 1
        parents = new_parents
    return G


def create_graph(probability_matrix, npseed=None):
    _rng = np.random.default_rng(npseed)
    G = nx.Graph()
    # randomprobability_matrix = self._rng.random((self.n_all,self.n_all))
    # it's better to use this function not rand_seed
    # https://albertcthomas.github.io/good-practices-random-number-generators/
    for i in range(np.shape(probability_matrix)[0]):
        G.add_node(i)
        for j in range(i):
            # weight = np.random.poisson(probability_matrix[i][j])
            if _rng.random() < probability_matrix[i][j] and i != j:
                G.add_edge(i, j)
    return G


def create_graph_fast(partition, sizes, p_matrix, npseed=None):
    _rng = np.random.default_rng(npseed)
    g = nx.Graph()
    g.graph["partition"] = partition
    # Setup nodes and graph name
    for block_id, nodes in enumerate(g.graph["partition"]):
        for node in nodes:
            g.add_node(node, block=block_id)

    g.name = "hierarchical stochastic_block_model"

    # Test for edge existence
    parts = partition
    block_range = range(len(sizes))
    block_iter = itertools.combinations_with_replacement(block_range, 2)
    for i, j in block_iter:
        if i == j:
            edges = itertools.combinations(parts[i], 2)
            for e in edges:
                if _rng.random() < p_matrix[i][j]:
                    g.add_edge(*e)
        else:
            edges = itertools.product(parts[i], parts[j])

        for e in edges:
            if _rng.random() < p_matrix[i][j]:
                g.add_edge(*e)  # __safe
    return g


class shsbm_nchild:
    def __init__(self, num_nodes, n_child, edge_density_per_layer):
        # edge_density_per_layer = [p0,p1,p2,p3,p4]
        self._num_nodes = num_nodes
        self._edge_density_per_layer = edge_density_per_layer
        self.n_child = n_child
        self.num_layer = len(edge_density_per_layer)
        self.num_communities = n_child ** (self.num_layer - 1)
        self.n_all = self._num_nodes * n_child ** (self.num_layer - 1)
        self.probability_matrix = self._edge_density_per_layer[0] * np.ones(
            (self.n_all, self.n_all)
        )
        for l, p in enumerate(self._edge_density_per_layer[1:]):
            n = self._num_nodes * n_child ** (self.num_layer - 2 - l)
            for ll in range(n_child ** (l + 1)):
                self.probability_matrix[(ll * n) : ((ll + 1) * n)].T[
                    (ll * n) : ((ll + 1) * n)
                ] = p * np.ones(n)
        self.tree_labels = self.tree_label()
        self.true_label = self.true_label()

    def create_graph(self, npseed=None):
        return create_graph(self.probability_matrix, npseed=npseed)

    def pos_flex(self, G, layout=nx.spring_layout):
        clusterings = self.true_clustering(self.num_layer - 2)
        pos_flex = positions.pos_flex(G, clusterings)
        return pos_flex

    def pos_hierarchy_nchild(self, G, layout=nx.spring_layout):
        return positions.pos_hierarchy_nchild(
            G, self._num_nodes, self.num_layer, self.n_child, layout=layout
        )

    def pos_for2(self, G, layout=nx.spring_layout):
        _num_layers = len(self._edge_density_per_layer) - 1
        return positions.pos(G, _num_layers, self._num_nodes, layout=layout)

    def pos_for2_with_noise(self, G, layout=nx.spring_layout, npseed=None):
        _num_layers = len(self._edge_density_per_layer) - 1
        return positions.pos_with_noise(
            G, _num_layers, self._num_nodes, layout=layout, npseed=npseed
        )

    # layer 1,2,3 not from 0
    def true_clustering_legacy(self, layer):
        n = self._num_nodes * self.n_child ** (self.num_layer - layer)
        true_clustering = [
            list(np.arange((ll * n), ((ll + 1) * n)))
            for ll in range(self.n_child ** (layer - 1))
        ]
        return true_clustering

    # layer 0, 1,2,3 from 0
    def true_clustering(self, layer):
        n = self._num_nodes * self.n_child ** (self.num_layer - layer - 1)
        true_clustering = [
            list(np.arange((ll * n), ((ll + 1) * n)))
            for ll in range(self.n_child ** (layer))
        ]
        return true_clustering

    def true_clustering_small_legacy(self, layer):
        n = 1 * self.n_child ** (self.num_layer - layer)
        true_clustering_small = [
            list(np.arange((ll * n), ((ll + 1) * n)))
            for ll in range(self.n_child ** (layer - 1))
        ]
        return true_clustering_small

    def true_clustering_small(self, layer):
        n = 1 * self.n_child ** (self.num_layer - layer - 1)
        true_clustering_small = [
            list(np.arange((ll * n), ((ll + 1) * n)))
            for ll in range(self.n_child ** (layer))
        ]
        return true_clustering_small

    def num_clusters_on_l(self, layer):
        return self.n_child**layer

    def calc_accuracy(
        self, estimated_clustering, layer, calc_acc_algo=mea.calc_accuracy
    ):
        true_clustering = self.true_clustering(layer)
        if len(estimated_clustering) < len(true_clustering):
            return np.nan
        else:
            return calc_acc_algo(estimated_clustering, true_clustering, self.n_all)[0]

    def true_label(self):
        true_label = np.zeros(self.n_all, dtype=int)
        for k in range(self.n_child ** (self.num_layer - 1)):
            true_label[k * self._num_nodes : (k + 1) * self._num_nodes] = k
        return true_label

    def true_St_old(self):
        St = np.ones((self.n_all, self.n_all), dtype=int) * (self.num_layer - 1)
        for l in range(self.num_layer):
            true_clustering = self.true_clustering(l)
            for cl in true_clustering:
                St = utild.smart_assigment(St, cl, cl, self.num_layer - 1 - l)
        return St

    def true_St_small_old(self):
        St_small = np.ones((self.num_communities, self.num_communities), dtype=int) * (
            self.num_layer - 1
        )
        for l in range(self.num_layer):
            true_clustering_small = self.true_clustering_small(l)
            for cl in true_clustering_small:
                St_small = utild.smart_assigment(
                    St_small, cl, cl, self.num_layer - 1 - l
                )
        return St_small

    def true_St(self):
        St = np.ones((self.n_all, self.n_all), dtype=int) * (self.num_layer - 1)
        for l in range(self.num_layer):
            true_clustering = self.true_clustering(l)
            for cl in true_clustering:
                St = utild.smart_assigment(St, cl, cl, l)
        return St

    def true_St_small(self):
        St_small = np.ones((self.num_communities, self.num_communities), dtype=int) * (
            self.num_layer - 1
        )
        for l in range(self.num_layer):
            true_clustering_small = self.true_clustering_small(l)
            for cl in true_clustering_small:
                St_small = utild.smart_assigment(St_small, cl, cl, l)
        return St_small

    def tree_label(self):
        tree_labels = np.zeros((self.n_all, self.num_layer - 1), dtype=int)
        for l in range(self.num_layer - 1):
            n = self._num_nodes * self.n_child ** (self.num_layer - 2 - l)
            for ll in range(self.n_child ** (l + 1)):
                tree_labels.T[l][(ll * n) : ((ll + 1) * n)] = ll % self.n_child
        return tree_labels

    def error_to_label0_legacy(self, p, npseed=None):
        _rng = np.random.default_rng(npseed)
        indexes = np.where(_rng.random(self.n_all) < p)[0]
        change_layers = _rng.integers(0, self.num_layer - 1, len(indexes))
        change_to = _rng.integers(1, self.n_child, len(indexes))
        tree_labels_with_error = self.tree_labels.copy()
        for ii, i in enumerate(indexes):
            tree_labels_with_error[i][change_layers[ii]] = (
                tree_labels_with_error[i][change_layers[ii]] + change_to[ii]
            ) % self.n_child
        return tree_labels_with_error

    def error_to_label0(self, p, npseed=None):
        _rng = np.random.default_rng(npseed)
        indexes = np.where(_rng.random(self.n_all) < p)[0]
        change_layers = _rng.integers(0, self.num_layer - 1, len(indexes))
        change_to = _rng.integers(1, self.n_child, len(indexes))
        tree_labels_with_error = self.tree_labels.copy()
        for ii, i in enumerate(indexes):
            tree_labels_with_error[i][change_layers[ii]] = (
                tree_labels_with_error[i][change_layers[ii]] + change_to[ii]
            ) % self.n_child
            tree_labels_with_error[i][
                min(change_layers[ii] + 1, self.num_layer - 1) :
            ] = _rng.integers(0, self.n_child, self.num_layer - 2 - change_layers[ii])
        return tree_labels_with_error

    def error_to_label1_legacy(self, p, layer, npseed=None):
        _rng = np.random.default_rng(npseed)
        indexes = np.where(_rng.random(self.n_all) < p)[0]
        change_to = _rng.integers(1, self.n_child, len(indexes))
        tree_labels_with_error = self.tree_labels.copy()
        for ii, i in enumerate(indexes):
            tree_labels_with_error[i][layer - 1] = (
                tree_labels_with_error[i][layer - 1] + change_to[ii]
            ) % self.n_child
        return tree_labels_with_error

    def error_to_label_to_bottom(self, p, npseed=None):
        _rng = np.random.default_rng(npseed)
        indexes = np.where(_rng.random(self.n_all) < p)[0]
        change_to = _rng.integers(1, self.n_child, len(indexes))
        tree_labels_with_error = self.tree_labels.copy()
        for ii, i in enumerate(indexes):
            tree_labels_with_error[i][-1] = (
                tree_labels_with_error[i][-1] + change_to[ii]
            ) % self.n_child
        return tree_labels_with_error

    def error_to_label1(self, p, layer, return_nonrandomized=False, npseed=None):
        _rng = np.random.default_rng(npseed)
        indexes = np.where(_rng.random(self.n_all) < p)[0]
        change_to = _rng.integers(1, self.n_child, len(indexes))
        tree_labels_with_error = self.tree_labels.copy()
        if return_nonrandomized:
            tree_labels_with_error_non_randomized = self.tree_labels.copy()
        for ii, i in enumerate(indexes):
            tree_labels_with_error[i][layer - 1] = (
                tree_labels_with_error[i][layer - 1] + change_to[ii]
            ) % self.n_child
            if return_nonrandomized:
                tree_labels_with_error_non_randomized[i][layer - 1] = (
                    tree_labels_with_error_non_randomized[i][layer - 1] + change_to[ii]
                ) % self.n_child
            tree_labels_with_error[i][min(layer, self.num_layer - 1) :] = _rng.integers(
                0, self.n_child, self.num_layer - 1 - layer
            )
        if return_nonrandomized:
            return tree_labels_with_error, tree_labels_with_error_non_randomized
        else:
            return tree_labels_with_error

    def error_to_label_uniform_rand(self, p, npseed=None):
        _rng = np.random.default_rng(npseed)
        indexes = np.where(_rng.random(self.n_all) < p)[0]
        tree_labels_with_error = self.tree_labels.copy()
        for ii, i in enumerate(indexes):
            tree_labels_with_error[i] = _rng.integers(
                0, self.n_child, self.num_layer - 1
            )
        return tree_labels_with_error

    def tree_label_to_label(self, tree_labels, layer=None):
        if layer is None:
            layer = self.num_layer - 1
        labels = np.sum(
            tree_labels[:, :layer]
            * np.array([self.n_child**l for l in range(layer)])[::-1],
            axis=1,
        )
        return labels

    def switch_communities_error(self, layer=None, num_switch=1, npseed=None):
        labels_on_the_layer = self.tree_label_to_label(self.tree_labels, layer=layer)
        num_clusters_in_the2 = len(np.unique(labels_on_the_layer)) // 2
        if num_clusters_in_the2 <= num_switch:
            print("number of switches are too large")
            raise ValueError
        _rng = np.random.default_rng(npseed)
        indexes_to_switch0 = []
        indexes_to_switch1 = []
        while len(np.unique(indexes_to_switch0)) < num_switch:
            indexes_to_switch0 = _rng.integers(0, num_clusters_in_the2, num_switch)
        while len(np.unique(indexes_to_switch1)) < num_switch:
            indexes_to_switch1 = _rng.integers(
                num_clusters_in_the2, 2 * num_clusters_in_the2, num_switch
            )
        labels_w_switch = np.zeros(len(labels_on_the_layer), dtype=int)
        for i in (
            set(np.arange(num_clusters_in_the2, 2 * num_clusters_in_the2))
            - set(indexes_to_switch1)
        ) | set(indexes_to_switch0):
            labels_w_switch[labels_on_the_layer == i] = 1
        return labels_w_switch


class hsbm:
    # specified_ps: this is to specify inter group edge densities
    # eg) [(ig, jg, p0), (ig, ig, p1)] specifies edge density between group ig and ij as p0, and edge density inside group ig as p1
    def __init__(
        self,
        sizes,
        group_matrix,
        edge_densities_per_group,
        specified_ps=None,
        prepare_truth=True,
    ):
        self.group_matrix = np.array(group_matrix)
        if len(sizes) != len(group_matrix):
            raise ValueError("'sizes' and 'group_matrix' do not match.")
        elif np.max(group_matrix) + 1 != len(edge_densities_per_group):
            raise ValueError(
                "'group number' and 'edge_densities_per_group' do not match."
            )
        elif (
            len(np.unique(group_matrix)) != np.max(group_matrix) + 1
            or np.min(group_matrix) != 0
        ):
            raise ValueError(
                "group_matrix has some missing number(s). group_matrix needs to contain serial integers starting from 0."
            )
        elif not np.all(self.group_matrix == self.group_matrix.T):
            raise ValueError("group_matrix needs to be symmetry.")
        for prob in edge_densities_per_group:
            if prob < 0 or prob > 1:
                raise ValueError("Entries of 'p' not in [0,1].")
        self.nodelist = range(0, sum(sizes))
        self.sizes = sizes
        self.edge_densities_per_group = edge_densities_per_group
        self.num_layer = len(edge_densities_per_group)
        self.num_communities = len(group_matrix)
        self.n_all = sum(sizes)
        # Split nodelist in a partition (list of sets).
        size_cumsum = [sum(self.sizes[0:x]) for x in range(0, len(self.sizes) + 1)]
        self.partition = [
            set(self.nodelist[size_cumsum[x] : size_cumsum[x + 1]])
            for x in range(0, len(size_cumsum) - 1)
        ]
        self.p_matrix = np.zeros(self.group_matrix.shape)
        for _g, _p in enumerate(edge_densities_per_group):
            self.p_matrix[self.group_matrix == _g] = _p
        if specified_ps is not None:
            for (ig, jg, p) in specified_ps:
                self.p_matrix[ig][jg] = p
                self.p_matrix[jg][ig] = p
        if prepare_truth:
            self.true_label = np.zeros(self.n_all, dtype=int)
            for block_id, nodes in enumerate(self.partition):
                for node in nodes:
                    self.true_label[node] = block_id
            self.probability_matrix = self.probability_matrix()

    def true_clustering_small_old(self, distance):
        block_range = range(len(self.sizes))
        groups_remaining = set(range(len(self.sizes)))
        clusterings = []
        group_d = self.group_matrix <= distance
        while len(groups_remaining) >= 1:
            ig = groups_remaining.pop()
            g = set(np.where(group_d[ig])[0])
            g = set([ig]) | g
            clusterings.append(g)
            groups_remaining = groups_remaining - g
        return clusterings

    def true_clustering_small(self, similarity):
        block_range = range(len(self.sizes))
        groups_remaining = set(range(len(self.sizes)))
        clusterings = []
        group_d = self.group_matrix >= similarity
        while len(groups_remaining) >= 1:
            ig = groups_remaining.pop()
            g = set(np.where(group_d[ig])[0])
            g = set([ig]) | g
            clusterings.append(g)
            groups_remaining = groups_remaining - g
        return clusterings

    def num_clusters_on_l(self, layer):
        similarity = layer
        return len(self.true_clustering_small(similarity))

    def true_clustering(self, similarity):
        groups = self.true_clustering_small(similarity)
        clusterings = [sum([list(self.partition[n]) for n in g], []) for g in groups]
        return clusterings

    def true_St(self):
        St = np.zeros((self.n_all, self.n_all), dtype=int)
        parts = self.partition
        block_range = range(len(self.sizes))
        block_iter = itertools.combinations_with_replacement(block_range, 2)
        for i, j in block_iter:
            iter = itertools.product(parts[i], parts[j])
            for it in iter:
                St[it[0]][it[1]] = self.group_matrix[i][j]
                St[it[1]][it[0]] = self.group_matrix[i][j]
        return np.array(St)

    def true_St_small(self):
        return self.group_matrix

    def probability_matrix(self):
        probability_matrix = np.zeros((self.n_all, self.n_all))
        parts = self.partition
        block_range = range(len(self.sizes))
        block_iter = itertools.combinations_with_replacement(block_range, 2)
        for i, j in block_iter:
            iter = itertools.product(parts[i], parts[j])
            for it in iter:
                probability_matrix[it[0]][it[1]] = self.p_matrix[i][j]
                probability_matrix[it[1]][it[0]] = self.p_matrix[i][j]
        return np.array(probability_matrix)

    def create_graph(self, npseed=None):
        _rng = np.random.default_rng(npseed)
        g = nx.Graph()
        g.graph["partition"] = self.partition
        # Setup nodes and graph name
        for block_id, nodes in enumerate(g.graph["partition"]):
            for node in nodes:
                g.add_node(node, block=block_id)

        g.name = "hierarchical stochastic_block_model"

        # Test for edge existence
        parts = self.partition
        block_range = range(len(self.sizes))
        block_iter = itertools.combinations_with_replacement(block_range, 2)
        for i, j in block_iter:
            if i == j:
                edges = itertools.combinations(parts[i], 2)
                for e in edges:
                    if _rng.random() < self.p_matrix[i][j]:
                        g.add_edge(*e)
            else:
                edges = itertools.product(parts[i], parts[j])

            for e in edges:
                if _rng.random() < self.p_matrix[i][j]:
                    g.add_edge(*e)  # __safe
        return g

    def pos_flex(self, G, layout=nx.spring_layout):
        pos, centers = positions.pos_flex(G, self.partition)
        return pos, centers

    def calc_accuracy(
        self, estimated_clustering, layer, calc_acc_algo=mea.calc_accuracy
    ):
        similarity = layer
        true_clustering = self.true_clustering(similarity)
        if len(estimated_clustering) < len(true_clustering):
            return np.nan
        else:
            return calc_acc_algo(estimated_clustering, true_clustering, self.n_all)[0]


class shsbm_nchild2(hsbm):
    def __init__(self, num_nodes, n_child, edge_density_per_layer, prepare_truth=True):
        num_layer = len(edge_density_per_layer)
        self.n_child = n_child
        group_matrix = utild.create_group_matrix(num_layer, num_child=n_child)
        edge_densities_per_group = edge_density_per_layer
        sizes = num_nodes * np.ones(n_child ** (num_layer - 1), dtype=int)
        hsbm.__init__(
            self, sizes, group_matrix, edge_densities_per_group, prepare_truth=prepare_truth, specified_ps=None
        )
        if prepare_truth:
            self.true_label = self.tree_label()

    def tree_label(self):
        tree_labels = np.zeros((self.n_all, self.num_layer - 1), dtype=int)
        for l in range(self.num_layer - 1):
            n = self._num_nodes * self.n_child ** (self.num_layer - 2 - l)
            for ll in range(self.n_child ** (l + 1)):
                tree_labels.T[l][(ll * n) : ((ll + 1) * n)] = ll % self.n_child
        return tree_labels

    def error_to_label0_legacy(self, p, npseed=None):
        _rng = np.random.default_rng(npseed)
        indexes = np.where(_rng.random(self.n_all) < p)[0]
        change_layers = _rng.integers(0, self.num_layer - 1, len(indexes))
        change_to = _rng.integers(1, self.n_child, len(indexes))
        tree_labels_with_error = self.tree_labels.copy()
        for ii, i in enumerate(indexes):
            tree_labels_with_error[i][change_layers[ii]] = (
                tree_labels_with_error[i][change_layers[ii]] + change_to[ii]
            ) % self.n_child
        return tree_labels_with_error

    def error_to_label0(self, p, npseed=None):
        _rng = np.random.default_rng(npseed)
        indexes = np.where(_rng.random(self.n_all) < p)[0]
        change_layers = _rng.integers(0, self.num_layer - 1, len(indexes))
        change_to = _rng.integers(1, self.n_child, len(indexes))
        tree_labels_with_error = self.tree_labels.copy()
        for ii, i in enumerate(indexes):
            tree_labels_with_error[i][change_layers[ii]] = (
                tree_labels_with_error[i][change_layers[ii]] + change_to[ii]
            ) % self.n_child
            tree_labels_with_error[i][
                min(change_layers[ii] + 1, self.num_layer - 1) :
            ] = _rng.integers(0, self.n_child, self.num_layer - 2 - change_layers[ii])
        return tree_labels_with_error

    def error_to_label1_legacy(self, p, layer, npseed=None):
        _rng = np.random.default_rng(npseed)
        indexes = np.where(_rng.random(self.n_all) < p)[0]
        change_to = _rng.integers(1, self.n_child, len(indexes))
        tree_labels_with_error = self.tree_labels.copy()
        for ii, i in enumerate(indexes):
            tree_labels_with_error[i][layer - 1] = (
                tree_labels_with_error[i][layer - 1] + change_to[ii]
            ) % self.n_child
        return tree_labels_with_error

    def error_to_label_to_bottom(self, p, npseed=None):
        _rng = np.random.default_rng(npseed)
        indexes = np.where(_rng.random(self.n_all) < p)[0]
        change_to = _rng.integers(1, self.n_child, len(indexes))
        tree_labels_with_error = self.tree_labels.copy()
        for ii, i in enumerate(indexes):
            tree_labels_with_error[i][-1] = (
                tree_labels_with_error[i][-1] + change_to[ii]
            ) % self.n_child
        return tree_labels_with_error

    def error_to_label1(self, p, layer, return_nonrandomized=False, npseed=None):
        _rng = np.random.default_rng(npseed)
        indexes = np.where(_rng.random(self.n_all) < p)[0]
        change_to = _rng.integers(1, self.n_child, len(indexes))
        tree_labels_with_error = self.tree_labels.copy()
        if return_nonrandomized:
            tree_labels_with_error_non_randomized = self.tree_labels.copy()
        for ii, i in enumerate(indexes):
            tree_labels_with_error[i][layer - 1] = (
                tree_labels_with_error[i][layer - 1] + change_to[ii]
            ) % self.n_child
            if return_nonrandomized:
                tree_labels_with_error_non_randomized[i][layer - 1] = (
                    tree_labels_with_error_non_randomized[i][layer - 1] + change_to[ii]
                ) % self.n_child
            tree_labels_with_error[i][min(layer, self.num_layer - 1) :] = _rng.integers(
                0, self.n_child, self.num_layer - 1 - layer
            )
        if return_nonrandomized:
            return tree_labels_with_error, tree_labels_with_error_non_randomized
        else:
            return tree_labels_with_error

    def error_to_label_uniform_rand(self, p, npseed=None):
        _rng = np.random.default_rng(npseed)
        indexes = np.where(_rng.random(self.n_all) < p)[0]
        tree_labels_with_error = self.tree_labels.copy()
        for ii, i in enumerate(indexes):
            tree_labels_with_error[i] = _rng.integers(
                0, self.n_child, self.num_layer - 1
            )
        return tree_labels_with_error

    def tree_label_to_label(self, tree_labels, layer=None):
        if layer is None:
            layer = self.num_layer - 1
        labels = np.sum(
            tree_labels[:, :layer]
            * np.array([self.n_child**l for l in range(layer)])[::-1],
            axis=1,
        )
        return labels

    def switch_communities_error(self, layer=None, num_switch=1, npseed=None):
        labels_on_the_layer = self.tree_label_to_label(self.tree_labels, layer=layer)
        num_clusters_in_the2 = len(np.unique(labels_on_the_layer)) // 2
        if num_clusters_in_the2 <= num_switch:
            print("number of switches are too large")
            raise ValueError
        _rng = np.random.default_rng(npseed)
        indexes_to_switch0 = []
        indexes_to_switch1 = []
        while len(np.unique(indexes_to_switch0)) < num_switch:
            indexes_to_switch0 = _rng.integers(0, num_clusters_in_the2, num_switch)
        while len(np.unique(indexes_to_switch1)) < num_switch:
            indexes_to_switch1 = _rng.integers(
                num_clusters_in_the2, 2 * num_clusters_in_the2, num_switch
            )
        labels_w_switch = np.zeros(len(labels_on_the_layer), dtype=int)
        for i in (
            set(np.arange(num_clusters_in_the2, 2 * num_clusters_in_the2))
            - set(indexes_to_switch1)
        ) | set(indexes_to_switch0):
            labels_w_switch[labels_on_the_layer == i] = 1
        return labels_w_switch
