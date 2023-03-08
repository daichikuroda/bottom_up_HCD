#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 17:20:31 2023

@author: kurodadaichi
"""

import urllib.request
import io
import zipfile

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score

import wrapper as wra
import handle_realdata as hr
import plots
import utild
import measurements as mea
import plots

exclude_independent = True
algos = ["rbu", "rbp", "paris"]  #  ["rbu", "rbp", "paris"]
num_samples = 1

G, txt = hr.create_football_net()
G, decode, encode = hr.change_node_nums(G)
G = hr.data_correction(G)
# print(txt)
# # print degree for each team - number of games
# for n, d in G.degree():
#     print(f"{n:20} {d:2}")

group_names = {
    0: "Atlantic Coast",
    1: "Big East",
    2: "Big Ten",
    3: "Big Twelve",
    4: "Conference USA",
    5: "Independents",
    6: "Mid-American",
    7: "Mountain West",
    8: "Pacific Ten",
    9: "Southeastern",
    10: "Sun Belt",
    11: "Western Athletic",
    12: "Big West",
}
conference_headquater = {
    0: ("Atlantic Coast", "North Carolina"),
    1: ("Big East", "Rhode Island"),
    2: ("Big Ten", "Illinois"),
    3: ("Big Twelve", "Texas"),
    4: ("Conference USA", "Texas"),
    6: ("Mid-American", "Ohio"),
    7: ("Mountain West", "Colorado"),
    8: ("Pacific Ten", "California"),
    9: ("Southeastern", "Alabama"),
    11: ("Western Athletic", "Texas"),
    12: ("Big West", "California"),
}

measurement_names = ["num community", "AMI", "ARI", "accuracy"]
measurements = {
    mea: {algo: np.zeros(num_samples) for algo in algos} for mea in measurement_names
}


if exclude_independent:
    nodes_list = [k for k, v in dict(G.nodes("value")).items() if v != 5]
    G = G.subgraph(nodes_list)
    G, decode2, encode2 = hr.change_node_nums(G)

groups = {}
hcs = {}
true_label = [v for v in dict(G.nodes(data="value")).values()]
for n, g in G.nodes(data="value"):
    _l = groups.get(group_names[g], [])
    _l.append(n)
    groups[group_names[g]] = _l
for isample in range(num_samples):
    for algo in algos:
        hcs[algo] = wra.hierarchical_communities(
            G,
            algo,
            metrics=[],
            weighted=False,
            num_communities=11,
        )
        for _mea, s in zip(
            measurement_names,
            [
                len(hcs[algo].communities),
                adjusted_mutual_info_score(true_label, hcs[algo].labelk),
                adjusted_rand_score(true_label, hcs[algo].labelk),
                mea.calc_accuracy(
                    list(groups.values()), hcs[algo].communitiesk, len(G.nodes())
                )[0],
            ],
        ):
            measurements[_mea][algo][isample] = s
for algo in algos:
    if algo == "bayesian":
        dG, _c = utild.nx_dendrogram(
            hcs[algo].community_bits,
            hcs[algo].bottom_communities,
            from_community_bits=True,
        )
    else:
        dG, _c = utild.nx_dendrogram(
            hcs[algo].Z,
            hcs[algo].bottom_communities,
        )
    _bool = algo == "paris"
    res = {"paris": 100, "rbu": 300, "bayesian": 20}
    dG = utild.dG_with_distance(
        dG,
        logscale=_bool,
        resolution=res.get(algo, 100),  # last_size=1 / 5
    )
    # nx.write_edgelist(dG, "test.edgelist", data=True)
    plots.draw_network_and_dendrogram(
        G,
        dG,
        groups,
        legend_size=6,
        node_size=15,
        edges_alpha=0.3,
        weights=500,
        # knot_size=_ks,
        marker_scale=1,
    )

for _mea in measurement_names:
    for _algo in algos:
        print(_mea, _algo, np.mean(measurements[_mea][_algo]))
for _algo in algos:
    _bool = _algo != "rbp"
    plots.plot_dendrogram(hcs[_algo].Z, logscale=(_bool))
    print(algo, "num community", len(hcs[algo].communities))
    print(algo, "ami", adjusted_mutual_info_score(true_label, hcs[algo].labelk))
    print(algo, "ari", adjusted_rand_score(true_label, hcs[algo].labelk))
    print(
        algo,
        "accuracy",
        mea.calc_accuracy(
            list(groups.values()), hcs[algo].communitiesk, len(G.nodes())
        )[0],
    )
