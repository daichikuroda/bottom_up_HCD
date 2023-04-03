#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import wrapper as wra
import plots
import utild

sys.path += [
    "./Paris/paris_codes/",
]
from paris import paris
import utils

# from pygenstability import run, plotting

powergrid_path = "./PyGenStability-master/examples/real_examples/powergrid/"
result_path = "./../results/newresults/powergrid2/"

algos = ["rbp", "rbu", "paris"]  # , "bayesian"]

edges = np.genfromtxt(powergrid_path + "UCTE_edges.txt")
location = np.genfromtxt(powergrid_path + "UCTE_nodes.txt")
posx = location[:, 1]
posy = location[:, 2]
pos = {}

edges = np.array(edges, dtype=np.int32)
G = nx.Graph()  # empty graph
G.add_edges_from(edges)  # add edges

# resetting label ids
G = nx.convert_node_labels_to_integers(G, label_attribute="old_label")

# updating label names and applying positions as attribute
for i in G.nodes():
    pos[i] = np.array(
        [posx[G.nodes[i]["old_label"] - 1], posy[G.nodes[i]["old_label"] - 1]]
    )
    G.nodes[i]["pos"] = pos[i].reshape(-1)

hcs = {}

for _algo in algos:
    hcs[_algo] = wra.hierarchical_communities(G, _algo)
for _algo in algos:
    add_params_plot = {
        "rbp": dict(
            clustering_point=utild.clustering_k_communities_by_similarities,
            similarities=hcs["rbp"].similarities,
        )
    }
    for num_cluster in range(1, min(len(hcs[_algo].bottom_communities) + 1, 40)):
        # for num_cluster in range(1, len(hcs[_algo].bottom_communities) + 1):
        plots.plot_for_recur_clustering(
            G,
            hcs[_algo].bottom_communities,
            hcs[_algo].Z,
            num_cluster,
            pos=pos,
            node_size=5,
            edge_width=0.5,  # ),
            save_path=result_path + _algo + str(num_cluster) + ".pdf",
            with_title=False,
            alpha=1.0,
            node_linew=0,
            **add_params_plot.get(_algo, dict())
        )

        if _algo == "rbp" or _algo == "bayesian":
            _bool = False
        else:
            _bool = True
    plots.plot_dendrogram(
        hcs[_algo].Z,
        save_path=result_path + _algo + "_dendrogram.pdf",
        logscale=_bool,
    )
    if _algo == "rbp":
        plots.plot_dendrogram_with_sim(
            hcs[_algo].Z,
            hcs[_algo].similarities,
            logscale=True,
            allow_negative=True,
            save_path=result_path + _algo + "_sim_dendrogram.pdf",
        )
