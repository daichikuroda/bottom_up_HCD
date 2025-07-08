#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: daichikuroda
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import HSBMs as hsbms
import measurements as mea
import utils as utils
import plots as plots
import bethe_hessian as cla
import recursive as rec
import synthesis as syn
from skbio.tree import TreeNode, upgma
from skbio import DistanceMatrix
import itertools as it
import edge_density_matrix as bmi


beta = 0.36

degrees = [d for d in range(5, 30, 5)]


num_nodes = 200
num_layer = 6
parent_seed = 13
num_samples = 10
num_child = 2
num_true_communities = 2 ** (num_layer - 1)
N = num_nodes * num_true_communities


algos = ["rbu", "synthesis"]

denominator = (1 - (2 * beta) ** (num_layer - 1)) / (1 - 2 * beta)
p_lasts = np.array(
    [2 ** (num_layer - 1) * d / ((1 + beta * denominator) * N) for d in degrees]
)


_rng = np.random.default_rng(parent_seed)
np_seeds = []
rf_scores = {algo: np.ones((len(p_lasts), num_samples)) for algo in algos}
baccs = {algo: np.zeros((len(p_lasts), num_samples)) for algo in algos}

for ip, p_last in enumerate(p_lasts):
    edge_densities = p_last * beta ** (np.arange(0, num_layer))[::-1]
    # a_list = a_lists[ip]
    if edge_densities[-1] > 1.0:
        print(edge_densities)
        raise ValueError
    print("edge_densities: ", edge_densities)
    # layers_to_see = list(np.arange(1, num_layer))
    shsbm_model = hsbms.shsbm_nchild(num_nodes, num_child, edge_densities)
    true_tree_D = utils.lca_to_distance_matrix(shsbm_model.group_matrix)
    B_true = shsbm_model.p_matrix
    P_true = shsbm_model.probability_matrix
    ground_truth = [list(g) for g in shsbm_model.partition]
    true_tree = upgma(DistanceMatrix(true_tree_D))
    St_true_small = shsbm_model.true_St_small()
    for iseed in range(num_samples):
        seed = _rng.integers(10**4)
        G = shsbm_model.create_graph(npseed=seed)
        print("expected degree: ", np.mean(list(dict(G.degree).values())))
        np_seeds.append(seed)
        A = nx.to_scipy_sparse_array(G, nodelist=sorted(G.nodes)).toarray()
        for algo in algos:
            if algo == "rbu":
                bottom_label, _zeta_p = cla.community_detection(
                    G, n_clusters=num_true_communities
                )
                bottom_communities = utils.return_communities(
                    bottom_label, np_nodes=np.array(sorted(G.nodes))
                )
                Z = rec.bottom_up(
                    G,
                    bottom_label,
                    linkage_algo="update_each",
                    sim_to_distance=None,
                )
                (
                    acc,
                    (ordered_truth, ordered_estimated),
                    _cm,
                    cm,
                ) = mea.calc_accuracy(
                    bottom_communities,
                    [list(g) for g in ground_truth],
                    different_size=True,
                )
                hB = bmi.calc_edge_density_matrix(A, bottom_communities)
                tree = TreeNode.from_linkage_matrix(Z, ordered_estimated)
            elif algo == "synthesis":
                n = A.shape[0]
                hDelta = A.sum() / n
                epsilon = 0.01 * hDelta**0.5 / n
                tree, hZ, hD = syn.synthesis(A, k=num_true_communities, epsilon=epsilon)
                hB = np.exp(-hD)
                bottom_label = np.argmax(hZ, axis=1)
                bottom_communities = utils.return_communities(
                    bottom_label, np_nodes=np.array(sorted(G.nodes))
                )
                (
                    acc,
                    (ordered_truth, ordered_estimated),
                    _cm,
                    cm,
                ) = mea.calc_accuracy(
                    bottom_communities,
                    [list(g) for g in ground_truth],
                    different_size=True,
                )
                ## relabel tree
                label_map = dict(zip(ordered_estimated, ordered_truth))
                for tip in tree.tips():
                    if int(tip.name) in label_map:
                        tip.name = str(label_map[int(tip.name)])
            baccs[algo][ip, iseed] = acc
            truth_to_est_map = dict(zip(ordered_truth, ordered_estimated))
            rf_scores[algo][ip, iseed] = true_tree.compare_rfd(tree, rooted=False) / (
                2 * num_true_communities - 3
            )

for algo in algos:
    print(f"RF scores for {algo}: ", rf_scores[algo])
    print(f"BACCs for {algo}: ", baccs[algo])


# To avoid the figures overflow
plt.rcParams["figure.subplot.bottom"] = 0.185
plt.rcParams["figure.subplot.left"] = 0.17

labels = {
    "rbu": "bottom-up",
    "synthesis": "synthesis",
    "rbp": "top-down",
}
num_for_algos = {"rbu": 3, "synthesis": 0, "rbp": 1}

for algo in algos:
    plt.errorbar(
        degrees,
        rf_scores[algo].mean(axis=1),
        yerr=rf_scores[algo].std(axis=1) / np.sqrt(num_samples),
        label=labels[algo],
        linestyle=utils.line_styles[num_for_algos[algo]],
        marker=utils.markers[num_for_algos[algo]],
        color=utils.colors[num_for_algos[algo]],
    )
plt.xlabel("expected degree", size=plots.axis_label_size)
plt.ylabel("Robinson-Foulds", size=plots.axis_label_size)
plt.xticks(
    degrees,
    fontsize=plots.tick_size,
)
plt.yticks(
    fontsize=plots.tick_size,
)
plt.ylim(-0.1, 1.1)
plt.legend(fontsize=plots.legend_font_size)
plt.title("RF scores")
plt.show()


for algo in algos:
    plt.errorbar(
        degrees,
        baccs[algo].mean(axis=1),
        yerr=baccs[algo].std(axis=1) / np.sqrt(num_samples),
        label=labels[algo],
        linestyle=utils.line_styles[num_for_algos[algo]],
        marker=utils.markers[num_for_algos[algo]],
        color=utils.colors[num_for_algos[algo]],
    )
plt.xlabel("expected degree", size=plots.axis_label_size)
plt.ylabel("accuracy", size=plots.axis_label_size)
plt.xticks(
    degrees,
    fontsize=plots.tick_size,
)
plt.yticks(
    fontsize=plots.tick_size,
)
plt.ylim(-0.1, 1.1)
plt.legend(fontsize=plots.legend_font_size)
plt.title("bottom accuracy")
plt.show()


for algo in algos:
    plt.errorbar(
        degrees,
        (rf_scores[algo] == 0).mean(axis=1),
        yerr=(rf_scores[algo] == 0).std(axis=1) / np.sqrt(num_samples),
        label=labels[algo],
        linestyle=utils.line_styles[num_for_algos[algo]],
        marker=utils.markers[num_for_algos[algo]],
        color=utils.colors[num_for_algos[algo]],
    )
plt.xlabel("expected degree", size=plots.axis_label_size)
plt.ylabel("recovery rate", size=plots.axis_label_size)
plt.xticks(
    degrees,
    fontsize=plots.tick_size,
)
plt.yticks(
    fontsize=plots.tick_size,
)
plt.ylim(-0.1, 1.1)
plt.legend(fontsize=plots.legend_font_size)
plt.title("tree recovery rate")
plt.show()
