#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 08:51:22 2022

@author: kurodadaichi
"""
import sys
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import handle_realdata as hr
import csv
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score
import utild
import recursive as rec
import beth_hessian as cla
import spectrald as spect
import plots
import measurements as mea
import positions
import wrapper as wra

sys.path.insert(0, "./Paris/paris_codes/")
from paris import paris
import utils

algos = ["rbu", "rbp", "paris"]
algo_names = {"rbu": "bottom-up", "rbp": "top-dwon"}

G = hr.high_school_net()
G, to_get_original_mapping, mapping = hr.change_node_nums(
    G
)  # graph tool conversion does not work without this procedure

# meta_data_f = high_school_folder + "metadata_2013.txt"
class_dict = {}
gender_dict = {}

class_dict = dict(G.nodes("class"))
class_names = set(class_dict.values())
classes = {k: [] for k in class_names}
classes = {k: v for (k, v) in sorted(classes.items())}  # for sorting
class_groups = ["2BIO", "PC", "MP", "PSI"]

for k, v in class_dict.items():
    classes[v].append(k)


true_label = np.zeros(len(G.nodes()), dtype=int)
nodes_num_map = dict(zip(G.nodes(), range(len(G.nodes()))))
for _cn, _m in enumerate(classes.values()):
    true_label[[nodes_num_map[_n] for _n in _m]] = _cn


add_params = {
    "bayesian": dict(
        edge_distribution="real-exponential",
        deg_corr=True,
    )
}
hcs = {}
for algo in algos:
    print(algo_names.get(algo, algo))
    hcs[algo] = wra.hierarchical_communities(
        G,
        algo,
        weighted=True,
        num_communities=9,
        continuous_node_label=False,
        # **add_params.get(algo, dict())
    )
    print("AMI: ", adjusted_mutual_info_score(true_label, hcs[algo].labelk))
    print("num community: ", len(hcs[algo].communities))

    detected_classes = utild.clustering_k_communities(
        hcs[algo].Z, 9, hcs[algo].bottom_communities
    )
    print(
        "accuracy: ",
        mea.calc_accuracy(
            list(classes.values()), hcs[algo].communitiesk, len(G.nodes())
        )[0],
    )
    print("______________________")

for algo in algos:
    name = algo_names.get(algo, algo)
    _bool = algo == "paris"
    plots.plot_dendrogram(hcs[algo].Z, logscale=_bool)
    plt.show()
    dG, _c = utild.nx_dendrogram(hcs[algo].Z, hcs[algo].bottom_communities)
    dG = utild.dG_with_distance(dG, logscale=_bool)
    nx.write_edgelist(dG, "test.edgelist", data=True)
    ns = {"rbp": 3}
    plots.draw_network_and_dendrogram(
        G,
        dG,
        classes,
        np.array(list(G.edges("weight"))),
        original_edges=True,
        knot_size=2,
        node_size=ns.get(algo, 4),
        legend_size=8,
    )

pos, centers = positions.pos_flex(
    G,
    list(classes.values()),
    scale=1 / 3,
)
for num_cluster in range(1, len(hcs["rbp"].bottom_communities) + 1):
    plots.plot_for_recur_clustering(
        G,
        hcs["rbp"].bottom_communities,
        hcs["rbp"].Z,
        num_cluster,
        pos=pos,
        node_size=20,
        edge_width=0.1 * np.array(list(G.edges("weight"))),
    )
