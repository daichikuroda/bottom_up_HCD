#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:43:54 2022

@author: maximilien
"""

import pandas as pd
import networkx as nx
from tqdm import tqdm


import numpy as np
import sys
from matplotlib import pyplot as plt
import wrapper as wra
import utild
import plots
import handle_realdata as hr

sys.path.insert(0, "./Paris/paris_codes/")

from paris import paris
import utils

import geopandas as gpd

import world_alliances as alli

algos = ["rbu", "rbp", "paris"]
algo_names = {
    "rbu": "bottom-up",
    "rbp": "top-down",
    "paris": "paris",
    "bayesian": "bayesian",
}

isolated_cluster = True

if __name__ == "__main__":

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = world[world.name != "Antarctica"]

    G = alli.makeMilitaryAllianceGraph(year=2018, alliance_types=["defense", "offense"])

    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    initial_communities_dict = {"rbp": [np.array(s.nodes()) for s in S]}
    hcs = {}
    countries = {}
    clusterings_dict = {}
    for node in G.nodes:
        countries[G.nodes[node]["iso_a3"]] = node

    for algo in algos:
        if isolated_cluster:
            hcs[algo] = wra.hierarchical_communities(
                S[0],
                algo,
                metrics=[],
                weighted=True,
            )
        else:
            hcs[algo] = wra.hierarchical_communities(
                G,
                algo,
                metrics=[],
                weighted=True,
                initial_communities=initial_communities_dict.get(algo, None),
            )

    for algo in algos:
        communities = hcs[algo].bottom_communities
        clusterings = []
        Z = hr.handle_inf_in_Z(hcs[algo].Z)
        if algo == "rbp" or algo == "bayesian":
            _bool = False
        else:
            _bool = True
        plots.plot_dendrogram(
            Z,
            logscale=_bool,
        )
        if algo == "paris":
            for clustering_rank in [1, 2, 3, 4]:
                clusterings.append(
                    utils.best_clustering(hcs["paris"].Z, k=clustering_rank)
                )
        else:
            for k in range(2, len(hcs[algo].bottom_communities) + 1):
                _cl = utild.clustering_k_communities(
                    hcs[algo].Z, k, hcs[algo].bottom_communities
                )
                if isolated_cluster:
                    _cl += [np.array(S[1].nodes())]
                clusterings.append(_cl)
        clusterings_dict[algo] = clusterings
        for l, communities in enumerate(clusterings):
            world[algo + str(l)] = np.nan
            communities_rest = []
            key_countries_labels_dict = dict(zip(alli.countries, alli.countries_nums))
            labels = {}
            allicountries = alli.countries.copy()
            key_countries_chosen = {}
            for _ic, _com in enumerate(communities):
                for key_country in allicountries:
                    # print(_ic, key_country)
                    if countries[key_country] in _com:
                        allicountries.remove(key_country)
                        labels[key_country] = key_countries_labels_dict.pop(key_country)
                        key_countries_chosen[key_country] = _com
                        break
                else:
                    communities_rest.append(_com)
            # print("_____________")
            for kc, key_community in key_countries_chosen.items():
                for _node in key_community:
                    world.loc[
                        world["iso_a3"] == G.nodes[_node]["iso_a3"], algo + str(l)
                    ] = labels[kc]

            labns = list(key_countries_labels_dict.values()) + list(
                range(7, len(communities))
            )
            for il, _com in zip(labns, communities_rest):
                for _node in _com:
                    world.loc[
                        world["iso_a3"] == G.nodes[_node]["iso_a3"], algo + str(l)
                    ] = il

    for algo in algos:
        for l in range(len(clusterings_dict[algo])):
            world = world.dropna()
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
            world.plot(ax=ax1, column=algo + str(l))
            ax1.set_axis_off()
            plt.show()
