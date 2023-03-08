#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:08:34 2022

@author: maximilien
"""

import pandas as pd
import networkx as nx
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

countries = ["FRA", "CHN", "RUS", "USA", "MLI", "COD", "DZA"]
countries_nums = [1, 6, 5, 0, 3, 4, 2]


def makeMilitaryAllianceGraph(year=2018, alliance_types=["defense"]):
    """
    alliance_type: atopally, defense, offense, neutral, nonagg, consul
    """

    dyad_data = pd.read_csv("military_alliances/atop5_1dy.csv")
    dyad_data = dyad_data.drop(
        columns=[
            "atopid1",
            "atopid2",
            "atopid3",
            "atopid4",
            "atopid5",
            "atopid6",
            "atopid7",
            "atopid8",
            "atopid9",
            "mem1",
            "mem2",
            "version",
        ]
    )
    dyad_data = dyad_data.loc[dyad_data["year"] == year]

    allied = dyad_data[alliance_types[0]] == 1
    for alliance_type in alliance_types:
        allied += dyad_data[alliance_type] == 1
    dyad_data = dyad_data.loc[allied]

    cow_countries = pd.read_csv("military_alliances/COW-country-codes.csv")

    cow_countries = cow_countries.dropna()
    cow_countries = cow_countries.drop_duplicates(keep="first")

    country_codes = cow_countries["CCode"].tolist()
    country_codes = set(country_codes)
    country_codes = list(country_codes)

    is_country_in_dataframe = [False] * len(country_codes)

    dyads = list(dyad_data["dyad"])
    for dyad in dyads:
        country1 = dyad % 1000
        country2 = (dyad - country1) // 1000
        if country1 in country_codes and country2 in country_codes:
            is_country_in_dataframe[country_codes.index(country1)] = True
            is_country_in_dataframe[country_codes.index(country2)] = True

    country_attributes = {}
    remaining_country_codes = []
    dummy = 0
    for code in country_codes:
        if is_country_in_dataframe[country_codes.index(code)]:
            filter = cow_countries["CCode"] == code
            data = cow_countries.where(filter).dropna()
            data_list = data.iloc[0].tolist()
            country_attributes[dummy] = {
                "SateAbbreviation": data_list[0],
                "CCode": int(data_list[1]),
                "StateName": data_list[2],
                "iso_a3": data_list[3],
            }

            remaining_country_codes.append(country_attributes[dummy]["CCode"])
            dummy += 1

    G = nx.Graph()
    G.add_nodes_from([i for i in range(len(country_attributes))])
    nx.set_node_attributes(G, country_attributes)

    for i in tqdm(range(len(dyad_data))):
        line = dyad_data.iloc[i]
        dyad = int(line["dyad"])

        country1 = dyad % 1000
        country2 = (dyad - country1) // 1000
        if country1 in remaining_country_codes and country2 in remaining_country_codes:
            G.add_edge(
                remaining_country_codes.index(country1),
                remaining_country_codes.index(country2),
            )

    return G


def plot_different_levels(
    world, communities, dendrogram, filename="military_alliances"
):
    n_communities = max(set(communities.values()))
    mega_communities = communities.copy()
    for level in range(dendrogram.shape[0]):
        communities_to_merge = [int(dendrogram[level, 0]), int(dendrogram[level, 1])]
        world, mega_communities = merge_communities(
            world,
            mega_communities,
            communities_to_merge,
            newCommunity=n_communities + 1 + level,
            newCommunitiesName="community",
        )
        world.plot(column="community")
        plt.savefig(filename + "_level_" + str(level) + ".eps")


def merge_communities(
    world,
    communities,
    communities_to_merge,
    newCommunity=0,
    newCommunitiesName="magacommunity",
):
    if newCommunity == 0:
        newCommunity = communities_to_merge[0]

    mega_communities = communities.copy()
    for country in communities.keys():
        if communities[country] in communities_to_merge:
            mega_communities[country] = newCommunity

    world[newCommunitiesName] = np.nan
    for country in mega_communities.keys():
        world.loc[world["iso_a3"] == country, newCommunitiesName] = mega_communities[
            country
        ]

    return world, mega_communities
