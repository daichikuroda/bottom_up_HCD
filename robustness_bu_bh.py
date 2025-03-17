#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:27:36 2025

@author: daichikuroda
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import HSBMs as hsbms
import measurements as mea
import wrapper as wra
import utild as utild
import plots as plots


result_path = "./../results/bottom_up/robust_from_bh2/"

diff = 0.025
betas = np.arange(0, 1.0 + diff, diff)
p_last = 0.08
num_nodes = 200
num_layer = 4
parent_seed = 13
num_samples = 200  # 25  # 10  # 25
num_child = 2
num_true_communities = 2 ** (num_layer - 1)
N = num_nodes * num_true_communities
layers_to_see = list(range(1, num_layer))


_rng = np.random.default_rng(parent_seed)

np_seeds = []
allocations_array = np.zeros((betas.shape[0], num_samples, 4))
bcorrefs = np.zeros((betas.shape[0], num_samples))
tree_correfs = np.zeros((betas.shape[0], num_samples))
tree_correfs2 = np.zeros((betas.shape[0], num_samples))

for ibeta, beta in enumerate(betas):
    edge_densities = p_last * beta ** (np.arange(0, num_layer))[::-1]
    # a_list = a_lists[ibeta]
    if edge_densities[-1] > 1.0:
        print(edge_densities)
        raise ValueError
    print("edge_densities: ", edge_densities)
    # layers_to_see = list(np.arange(1, num_layer))
    shsbm_model = hsbms.shsbm_nchild(num_nodes, num_child, edge_densities)
    true_tree_D = utild.lca_to_distance_matrix(shsbm_model.group_matrix)
    P_true = shsbm_model.probability_matrix
    for iseed in range(num_samples):
        seed = _rng.integers(10**4)
        G = shsbm_model.create_graph(npseed=seed)
        np_seeds.append(seed)
        St_true_small = shsbm_model.true_St_small()
        P_true = shsbm_model.probability_matrix
        rbu = wra.hierarchical_communities(
            G, "rbu", num_communities=num_true_communities
        )
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

        supercoms, _pm = utild.return_supercoms(rbu.Z, rbu.bottom_communities)
        com_bits = utild.supercoms_to_community_bits(supercoms)
        bcorrefs[ibeta][iseed] = mea.calc_corref_in_D(
            np.diag(np.ones(len(ground_truth))),
            np.diag(np.ones(len(rbu.bottom_communities))),
            [list(g) for g in ground_truth],
            rbu.bottom_communities,
        )
        tree_correfs[ibeta][iseed] = mea.calc_corref_in_D(
            true_tree_D,
            utild.lca_to_distance_matrix(mea.depth_lcas(com_bits, return_dict=False)),
            [list(g) for g in ground_truth],
            rbu.bottom_communities,
        )
        # tree_correfs2[ibeta][iseed] = mea.calc_corref_in_D(
        #     true_tree_D,
        #     utild.lca_to_distance_matrix(mea.depth_lcas(com_bits, return_dict=False)),
        #     [rbu.bottom_communities[_i] for _i in ordered_estimated],
        #     rbu.bottom_communities,
        # )
        tree_correfs2[ibeta][iseed] = np.corrcoef(
            true_tree_D.flatten(),
            utild.lca_to_distance_matrix(mea.depth_lcas(com_bits, return_dict=False))[
                ordered_estimated, :
            ][:, ordered_estimated].flatten(),
        )[0, 1]

        allocations = [
            cm[shsbm_model.group_matrix == i].mean()
            for i in np.unique(shsbm_model.group_matrix)
        ]
        allocations_array[ibeta][iseed] = allocations

np.savez(result_path + "allocation_array.npz", allocations_array)
np.savetxt(result_path + "tree_corref.csv", tree_correfs, delimiter=",")
np.savetxt(result_path + "tree_corref2.csv", tree_correfs2, delimiter=",")
np.savetxt(result_path + "bottom_corref.csv", bcorrefs, delimiter=",")
np.savetxt(result_path + "np_seeds.csv", np_seeds, delimiter=",", fmt="%i")


allocations_array = allocations_array / num_nodes
# To avoid the figures overflow
plt.rcParams["figure.subplot.bottom"] = 0.185
plt.rcParams["figure.subplot.left"] = 0.15
for j in range(4):
    plt.errorbar(
        betas,
        allocations_array.mean(axis=1)[:, j],
        yerr=allocations_array.std(axis=1, ddof=1)[:, j] / np.sqrt(num_samples),
        label=r"$\hat{\zeta}$(" + str(j) + ")",
        linestyle=utild.line_styles[j],
        marker=utild.markers[j],
        color=utild.colors[j],
    )
plt.xlabel(r"$\beta$", size=plots.axis_label_size)
plt.ylabel("accuracy", size=plots.axis_label_size)
plt.xticks(
    fontsize=plots.tick_size,
)
plt.yticks(
    fontsize=plots.tick_size,
)
plt.legend(fontsize=plots.legend_font_size)
plt.savefig(result_path + "lineplot.pdf")
plt.show()


for j in range(4):
    plt.errorbar(
        betas,
        allocations_array.mean(axis=1)[:, j]
        * ((shsbm_model.group_matrix == j).sum() / num_true_communities),
        yerr=allocations_array.std(axis=1, ddof=1)[:, j]
        * ((shsbm_model.group_matrix == j).sum() / num_true_communities) ** 2
        / np.sqrt(num_samples),
        label=r"$\hat{\zeta}$(" + str(j) + ")",
        linestyle=utild.line_styles[j],
        marker=utild.markers[j],
        color=utild.colors[j],
    )
plt.xlabel(r"$\beta$", size=plots.axis_label_size)
plt.ylabel("accuracy", size=plots.axis_label_size)
plt.xticks(
    fontsize=plots.tick_size,
)
plt.yticks(
    fontsize=plots.tick_size,
)
plt.legend(fontsize=plots.legend_font_size)
plt.savefig(result_path + "lineplot2.pdf")
plt.show()


plt.errorbar(
    betas,
    bcorrefs.mean(axis=1),
    yerr=bcorrefs.std(axis=1) / np.sqrt(num_samples),
    label="bottom",
    linestyle=utild.line_styles[3],
    marker=utild.markers[3],
    color=utild.colors[3],
)
plt.errorbar(
    betas,
    tree_correfs.mean(axis=1),
    yerr=tree_correfs.std(axis=1) / np.sqrt(num_samples),
    label="tree",
    linestyle=utild.line_styles[0],
    marker=utild.markers[4],
    color=utild.colors[5],
)
plt.xlabel(r"$\beta$", size=plots.axis_label_size)
plt.ylabel("correlation", size=plots.axis_label_size)
plt.xticks(
    fontsize=plots.tick_size,
)
plt.yticks(
    fontsize=plots.tick_size,
)
plt.legend(fontsize=plots.legend_font_size)
plt.savefig(result_path + "tree_corref.pdf")
plt.show()

plt.errorbar(
    betas,
    bcorrefs.mean(axis=1),
    yerr=bcorrefs.std(axis=1) / np.sqrt(num_samples),
    label="bottom",
    linestyle=utild.line_styles[3],
    marker=utild.markers[3],
    color=utild.colors[3],
)
plt.errorbar(
    betas,
    tree_correfs2.mean(axis=1),
    yerr=tree_correfs2.std(axis=1) / np.sqrt(num_samples),
    label="tree",
    linestyle=utild.line_styles[0],
    marker=utild.markers[4],
    color=utild.colors[5],
)
plt.xlabel(r"$\beta$", size=plots.axis_label_size)
plt.ylabel("correlation", size=plots.axis_label_size)
plt.xticks(
    fontsize=plots.tick_size,
)
plt.yticks(
    fontsize=plots.tick_size,
)
plt.legend(fontsize=plots.legend_font_size)
plt.savefig(result_path + "robustness.pdf")
plt.show()

plt.errorbar(
    betas,
    (tree_correfs2 >= 0.9999).mean(axis=1),
    yerr=tree_correfs2.std(axis=1) / np.sqrt(num_samples),
    label="tree",
    linestyle=utild.line_styles[0],
    marker=utild.markers[4],
    color=utild.colors[5],
)
plt.xlabel(r"$\beta$", size=plots.axis_label_size)
plt.ylabel("success rate", size=plots.axis_label_size)
plt.xticks(
    fontsize=plots.tick_size,
)
plt.yticks(
    fontsize=plots.tick_size,
)
plt.legend(fontsize=plots.legend_font_size)
plt.savefig(result_path + "tree_recovery_rate.pdf")
plt.show()


for ibeta, beta in enumerate(betas):
    if beta not in [0.4, 0.65]:
        continue
    edge_densities = p_last * beta ** (np.arange(0, num_layer))[::-1]
    shsbm_model = hsbms.shsbm_nchild(num_nodes, num_child, edge_densities)
    true_tree_D = utild.lca_to_distance_matrix(shsbm_model.group_matrix)
    P_true = shsbm_model.probability_matrix
    seed = np_seeds[num_samples * ibeta]
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
    # sns.set(font_scale=1.2)
    res = sns.heatmap(
        data=cm,  # (cm/np.sum(cm,axis=0)).T,
        # xticklabels=ordered_truth,
        xticklabels=["000", "001", "010", "011", "100", "101", "110", "111"],
        # yticklabels=np.hstack(
        #     (
        #         ordered_estimated,
        #         np.setdiff1d(np.arange(len(rbu.bottom_communities)), ordered_estimated),
        #     )
        # ),
        yticklabels=["000", "001", "010", "011", "100", "101", "110", "111"],
        annot=True,
        cmap="gist_yarg",
        annot_kws={"size": 15},
        fmt="g",
    )
    cbar = res.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize=15, rotation=270)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize=15, rotation=0)
    plt.savefig(result_path + str(beta) + "cm.pdf")
    plt.show()
