#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:27:36 2025

@author: daichikuroda
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import HSBMs as hsbms
import measurements as mea
import wrapper as wra
import utils as utils
import plots as plots
import bethe_hessian as cla
import recursive as rec


beta = 0.3

degrees = [d for d in range(5, 50, 5)]


num_nodes = 200
num_layer = 4
parent_seed = 13
parent_seed2 = 42
num_samples = 10
num_child = 2
num_true_communities = 2 ** (num_layer - 1)
N = num_nodes * num_true_communities
layers_to_see = list(range(1, num_layer))

split_probs = [0.9, 0.95, None]

denominator = (1 - (2 * beta) ** (num_layer - 1)) / (1 - 2 * beta)
p_lasts = np.array(
    [2 ** (num_layer - 1) * d / ((1 + beta * denominator) * N) for d in degrees]
)


_rng = np.random.default_rng(parent_seed)
_rng2 = np.random.default_rng(parent_seed2)
np_seeds = []

allocations_array_dict = {
    k: np.zeros((p_lasts.shape[0], num_samples, 4)) for k in split_probs
}
bcorrefs = {k: np.zeros((p_lasts.shape[0], num_samples)) for k in split_probs}
tree_correfs = {k: np.zeros((p_lasts.shape[0], num_samples)) for k in split_probs}
tree_correfs2 = {k: np.zeros((p_lasts.shape[0], num_samples)) for k in split_probs}


def graph_split(G, q=0.9, rng=_rng2):
    A = nx.adjacency_matrix(G, nodelist=np.array(sorted(G.nodes)))
    reveal = _rng.random(A.shape)
    A1 = (reveal <= q) * A
    A2 = (reveal > q) * A
    return A1, A2


for ip, p_last in enumerate(p_lasts):
    edge_densities = p_last * beta ** (np.arange(0, num_layer))[::-1]

    if edge_densities[-1] > 1.0:
        print(edge_densities)
        raise ValueError
    print("edge_densities: ", edge_densities)

    shsbm_model = hsbms.shsbm_nchild(num_nodes, num_child, edge_densities)
    true_tree_D = utils.lca_to_distance_matrix(shsbm_model.group_matrix)
    P_true = shsbm_model.probability_matrix
    for iseed in range(num_samples):
        seed = _rng.integers(10**4)
        G = shsbm_model.create_graph(npseed=seed)
        print("expected degree: ", np.mean(list(dict(G.degree).values())))
        np_seeds.append(seed)
        St_true_small = shsbm_model.true_St_small()
        P_true = shsbm_model.probability_matrix
        for q in split_probs:
            if q is None:
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
            else:
                A1, A2 = graph_split(G, q=q, rng=_rng2)
                bottom_label, _zeta_p = cla.community_detection(
                    A1, inputA=True, n_clusters=num_true_communities
                )
                bottom_communities = utils.return_communities(
                    bottom_label, np_nodes=np.array(sorted(G.nodes))
                )
                Z = rec.bottom_up(
                    A2,
                    bottom_label,
                    linkage_algo="update_each",
                    sim_to_distance=None,
                    inputA=True,
                )

            ground_truth = [list(g) for g in shsbm_model.partition]
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

            supercoms, _pm = utils.return_supercoms(Z, bottom_communities)
            com_bits = utils.supercoms_to_community_bits(supercoms)
            bcorrefs[q][ip][iseed] = mea.calc_corref_in_D(
                np.diag(np.ones(len(ground_truth))),
                np.diag(np.ones(len(bottom_communities))),
                [list(g) for g in ground_truth],
                bottom_communities,
            )
            tree_correfs[q][ip][iseed] = mea.calc_corref_in_D(
                true_tree_D,
                utils.lca_to_distance_matrix(
                    mea.depth_lcas(com_bits, return_dict=False)
                ),
                [list(g) for g in ground_truth],
                bottom_communities,
            )

            tree_correfs2[q][ip][iseed] = np.corrcoef(
                true_tree_D.flatten(),
                utils.lca_to_distance_matrix(
                    mea.depth_lcas(com_bits, return_dict=False)
                )[ordered_estimated, :][:, ordered_estimated].flatten(),
            )[0, 1]

            allocations = [
                cm[shsbm_model.group_matrix == i].mean()
                for i in np.unique(shsbm_model.group_matrix)
            ]
            allocations_array_dict[q][ip][iseed] = allocations

for q in split_probs:
    allocations_array = allocations_array_dict[q] / num_nodes
# To avoid the figures overflow
plt.rcParams["figure.subplot.bottom"] = 0.185
plt.rcParams["figure.subplot.left"] = 0.17


for q in split_probs:
    for j in range(4):
        plt.errorbar(
            degrees,
            allocations_array.mean(axis=1)[:, j]
            * ((shsbm_model.group_matrix == j).sum() / num_true_communities),
            yerr=allocations_array.std(axis=1, ddof=1)[:, j]
            * ((shsbm_model.group_matrix == j).sum() / num_true_communities) ** 2
            / np.sqrt(num_samples),
            label=r"$\hat{\zeta}$(" + str(j) + ")",
            linestyle=utils.line_styles[j],
            marker=utils.markers[j],
            color=utils.colors[j],
        )
    plt.xlabel(r"expected degree", size=plots.axis_label_size)
    plt.ylabel(r"$\hat{\zeta}$", size=plots.axis_label_size)
    plt.xticks(
        fontsize=plots.tick_size,
    )
    plt.yticks(
        fontsize=plots.tick_size,
    )
    plt.legend(fontsize=plots.legend_font_size)
    plt.title("lineplot with q: " + str(q))
    plt.show()

for q in split_probs:
    plt.errorbar(
        degrees,
        bcorrefs[q].mean(axis=1),
        yerr=bcorrefs[q].std(axis=1) / np.sqrt(num_samples),
        label="bottom",
        linestyle=utils.line_styles[3],
        marker=utils.markers[3],
        color=utils.colors[3],
    )
    plt.errorbar(
        degrees,
        tree_correfs[q].mean(axis=1),
        yerr=tree_correfs[q].std(axis=1) / np.sqrt(num_samples),
        label="tree",
        linestyle=utils.line_styles[0],
        marker=utils.markers[4],
        color=utils.colors[5],
    )
    plt.xlabel(r"expected degree", size=plots.axis_label_size)
    plt.ylabel("correlation", size=plots.axis_label_size)
    plt.xticks(
        fontsize=plots.tick_size,
    )
    plt.yticks(
        fontsize=plots.tick_size,
    )
    plt.legend(fontsize=plots.legend_font_size)
    plt.title("tree_corref with q: " + str(q))
    plt.show()

    plt.errorbar(
        degrees,
        bcorrefs[q].mean(axis=1),
        yerr=bcorrefs[q].std(axis=1) / np.sqrt(num_samples),
        label="bottom",
        linestyle=utils.line_styles[3],
        marker=utils.markers[3],
        color=utils.colors[3],
    )
    plt.errorbar(
        degrees,
        tree_correfs2[q].mean(axis=1),
        yerr=tree_correfs2[q].std(axis=1) / np.sqrt(num_samples),
        label="tree",
        linestyle=utils.line_styles[0],
        marker=utils.markers[4],
        color=utils.colors[5],
    )
    plt.xlabel(r"expected degree", size=plots.axis_label_size)
    plt.ylabel("correlation", size=plots.axis_label_size)
    plt.xticks(
        fontsize=plots.tick_size,
    )
    plt.yticks(
        fontsize=plots.tick_size,
    )
    plt.legend(fontsize=plots.legend_font_size)
    plt.title("robustness with q: " + str(q))
    plt.show()

    plt.errorbar(
        degrees,
        (tree_correfs2[q] >= 0.9999).mean(axis=1),
        yerr=tree_correfs2[q].std(axis=1) / np.sqrt(num_samples),
        label="tree",
        linestyle=utils.line_styles[0],
        marker=utils.markers[4],
        color=utils.colors[5],
    )
    plt.xlabel(r"expected degree", size=plots.axis_label_size)
    plt.ylabel("recovery rate", size=plots.axis_label_size)
    plt.xticks(
        fontsize=plots.tick_size,
    )
    plt.yticks(
        fontsize=plots.tick_size,
    )
    plt.legend(fontsize=plots.legend_font_size)
    plt.title("tree_recovery_rate with q: " + str(q))
    plt.show()

## recovery rate summary
plt.errorbar(
    degrees,
    (tree_correfs2[None] >= 0.9999).mean(axis=1),
    yerr=tree_correfs2[None].std(axis=1) / np.sqrt(num_samples),
    label="without",
    linestyle=utils.line_styles[1],
    marker=utils.markers[5],
    color=utils.colors[6],
)
plt.errorbar(
    degrees,
    (tree_correfs2[0.9] >= 0.9999).mean(axis=1),
    yerr=tree_correfs2[0.9].std(axis=1) / np.sqrt(num_samples),
    label="0.9",
    linestyle=utils.line_styles[0],
    marker=utils.markers[4],
    color=utils.colors[5],
)
plt.errorbar(
    degrees,
    (tree_correfs2[0.95] >= 0.9999).mean(axis=1),
    yerr=tree_correfs2[0.95].std(axis=1) / np.sqrt(num_samples),
    label="0.95",
    linestyle=utils.line_styles[2],
    marker=utils.markers[5],
    color=utils.colors[7],
)
plt.xlabel(r"expected degree", size=plots.axis_label_size)
plt.ylabel("recovery rate", size=plots.axis_label_size)
plt.xticks(
    fontsize=plots.tick_size,
)
plt.yticks(
    fontsize=plots.tick_size,
)
plt.legend(fontsize=plots.legend_font_size)
plt.title("tree_recovery_rate_bd.pdf")
plt.show()


for ip, p_last in enumerate(p_lasts):
    edge_densities = p_last * beta ** (np.arange(0, num_layer))[::-1]
    shsbm_model = hsbms.shsbm_nchild(num_nodes, num_child, edge_densities)
    true_tree_D = utils.lca_to_distance_matrix(shsbm_model.group_matrix)
    P_true = shsbm_model.probability_matrix
    seed = np_seeds[num_samples * ip]
    G = shsbm_model.create_graph(npseed=seed)
    rbu = wra.hierarchical_communities(G, "rbu", num_communities=num_true_communities)
    ground_truth = [list(g) for g in shsbm_model.partition]
    (
        acc,
        (ordered_truth, ordered_estimated),
        _cm,
        cm,
    ) = mea.calc_accuracy(
        rbu.bottom_communities,
        [list(g) for g in ground_truth],
        different_size=True,
    )
    sns.heatmap(
        data=cm,
        xticklabels=ordered_truth,
        yticklabels=ordered_truth,
        annot=True,
        cmap="gist_yarg",
        annot_kws={"size": 8},
        fmt="g",
    )
    plt.title(str(degrees[ip]) + "cm.pdf")
    plt.show()
