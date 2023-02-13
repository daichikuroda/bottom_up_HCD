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

import utild

legend_font_size = 18
tick_size = 18
axis_label_size = 24


def draw_network_and_dendrogram(
    G,
    dG,
    groups,
    weights=500,
    save_path=None,
    original_edges=True,
    node_size=3,
    edges_alpha=0.3,
    knot_size=3,  # 4,
    legend_on=True,
    legend_size=None,
    marker_scale=3,
):
    if legend_size is None:
        legend_size = legend_font_size
    group_colors = utild.colors[:6] + utild.colors[8:]
    tree_nodes0 = np.array(dG.nodes)
    between_nodes = tree_nodes0[tree_nodes0 <= -1]
    tree_nodes = tree_nodes0[
        tree_nodes0 >= 10 ** len(str(max(G.nodes)))
    ]  # tree_nodes[tree_nodes >= (max(G.nodes))]
    pos = nx.nx_agraph.graphviz_layout(dG, prog="twopi", args="", root=max(dG.nodes()))
    if original_edges:
        nx.draw_networkx_edges(
            G,
            width=0.0005 * weights,
            arrowstyle="-",
            pos=pos,
            edge_color="gray",
            alpha=edges_alpha,
        )
    nx.draw_networkx_edges(
        dG, width=0.7, arrows=False, pos=pos  # edge_color="gray"
    )  # arrowstyle="-"
    for i, (_cn, _cmembs) in enumerate(groups.items()):
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            node_size=node_size,
            nodelist=set(_cmembs) & set(dG.nodes),
            node_color=group_colors[i],
            label=_cn,
        )
    nx.draw_networkx_nodes(
        dG,
        nodelist=sorted(tree_nodes)[:-1],
        pos=pos,
        node_size=knot_size,
        node_shape="s",
        node_color="black",  # "gray"
    )
    nx.draw_networkx_nodes(
        dG,
        nodelist=[max(tree_nodes)],
        pos=pos,
        node_size=80,
        node_shape="*",
        node_color="black",  # "gray"
    )
    if legend_on:
        plt.legend(
            loc="upper right",
            bbox_to_anchor=(1.15, 1.0),
            frameon=False,
            markerscale=marker_scale,
            fontsize=legend_size,
        )
    plt.axis("off")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def heatmapsh_scatter(X, Y, Z, vmin=0, vmax=1, distinguish_zeros=False):
    if distinguish_zeros:
        zeros = Z.flatten() == 0
        plt.scatter(
            X.flatten()[zeros],
            Y.flatten()[zeros],
            c=Z.flatten()[zeros],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            marker="s",
        )
        plt.scatter(
            X.flatten()[zeros == False],
            Y.flatten()[zeros == False],
            c=Z.flatten()[zeros == False],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
    else:
        plt.scatter(
            X.flatten(),
            Y.flatten(),
            c=Z.flatten(),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=tick_size)


# Plot dendrogram
def plot_dendrogram(
    D,
    logscale=True,
    save_path=False,
    color_threshold=0,
    above_threshold_color="k",
    resize=True,
):
    if resize:
        plt.figure(figsize=(25, 10))
    Dlog = D.copy()
    if logscale:
        Dlog[:, 2] = np.log(Dlog[:, 2])
        Dlog[1:, 2] = Dlog[1:, 2] - Dlog[1, 2]
        Dlog[0, 2] = 0
    if color_threshold is None and above_threshold_color is None:
        sch.dendrogram(
            Dlog,
            leaf_rotation=90.0,
            # link_color_func=(1, 1, 1),
        )
    else:
        sch.dendrogram(
            Dlog,
            leaf_rotation=90.0,
            color_threshold=color_threshold,
            above_threshold_color="k",
            # link_color_func=(1, 1, 1),
        )
    plt.axis("off")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_dendrogram_from_group_matrix(
    group_matrix,
    logscale=False,
    save_path=False,
    color_threshold=0,
    above_threshold_color="k",
    new_group_matrix=True,
    resize=True,
):
    if new_group_matrix is True:
        group_matrix = np.max(group_matrix) + 1 - group_matrix
    D = sch.linkage(
        utild.distance_matrix_to_condensed_one(group_matrix), method="single"
    )
    plot_dendrogram(
        D,
        logscale=logscale,
        save_path=save_path,
        color_threshold=color_threshold,
        above_threshold_color=above_threshold_color,
        resize=resize,
    )


def plot_for_recur_clustering(
    G,
    communities,
    rbp,
    k,
    pos=nx.spring_layout,
    clustering_point=utild.clustering_k_communities,
    save_path=False,
    original_edges=True,
    width=16,
    height=8,
    node_size=1,
    edge_width=10,
):

    plt.rcParams.update({"font.size": 13})
    # plt.figure(figsize=(width, height))
    # plt.subplots_adjust(
    #     left=0.02, right=0.98, bottom=0.06, top=0.85, wspace=0.05, hspace=0.05
    # )
    clustering = clustering_point(rbp, k, communities)
    length = [len(c) for c in clustering]
    index = np.argsort(-np.array(length))
    plt.subplot(1, 1, 1)
    plt.axis("off")
    plt.title("k: " + str(k) + "\n(#clusters=" + str(len(clustering)) + ")")
    draw_nodes = nx.draw_networkx_nodes(
        G, pos, node_size=node_size, linewidths=node_size * 0.01, node_color="w"
    )
    draw_nodes.set_edgecolor("k")
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.1, edge_color="gray")
    nodes = list(G.nodes())
    for l in range(min(len(clustering), len(utild.colors))):
        nodelist = clustering[index[l]]  # [nodes[i] for i in clustering[index[l]]]
        draw_nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_size,
            linewidths=node_size * 0.01,
            nodelist=nodelist,
            node_color=utild.colors[l],
        )
        draw_nodes.set_edgecolor("k")
    if save_path:
        plt.savefig(save_path)
    plt.show()
