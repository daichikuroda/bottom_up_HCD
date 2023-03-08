#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""
import numpy as np
import networkx as nx
import sys
import graph_tool.all as gt

import util_for_gt as uti_gt
import utild

rec_type_list = [
    "real-exponential",
    "discrete-geometric",
    "discrete-binomial",
    "discrete-poisson",
]


def clustering_from_gtG(
    gtG, deg_corr=True, weighted=False, edge_distribution="best", np_nodes=None
):
    if weighted:
        if edge_distribution == "best":
            states = []
            for rec_type in rec_type_list:
                states.append(
                    gt.minimize_nested_blockmodel_dl(
                        gtG,
                        state_args=dict(
                            deg_corr=deg_corr,
                            recs=[gtG.ep.weight],
                            rec_types=[rec_type],
                        ),
                    )
                )
                _description_lengths = [
                    _state.entropy()  # + np.log(gtG.ep.weight.a).sum() this value is same over the distribution?
                    for _state in states
                ]
            _best = np.argmin(_description_lengths)
            print("description legth: ", _description_lengths, rec_type_list[_best])
            state = states[_best]
        else:
            state = gt.minimize_nested_blockmodel_dl(
                gtG,
                state_args=dict(
                    deg_corr=deg_corr,
                    recs=[gtG.ep.weight],
                    rec_types=[edge_distribution],
                ),
            )
    else:
        state = gt.minimize_nested_blockmodel_dl(
            gtG, state_args=dict(deg_corr=deg_corr)
        )
    bottom_block = state.project_level(0).get_blocks()
    bottom_label = [n for n in bottom_block]
    # bottom_communities = utild.return_communities(bottom_label, np_nodes=np_nodes)
    labels_levels = [bottom_label]
    for l in range(1, len(state.get_levels())):
        state2 = state.project_level(l)
        next_labels = [n for n in state2.get_blocks()]
        if next_labels != labels_levels[-1]:
            labels_levels.append(next_labels)
    community_bits = np.unique(np.array(labels_levels, dtype=int).T, axis=0)[:, ::-1]
    communities = utild.return_communities(
        bottom_label, np_nodes=gtG.get_vertices(), return_as_dict=True
    )
    communities = [communities[_l] for _l in community_bits.T[-1]]
    community_bits = [list(_cb) for _cb in community_bits]
    return communities, community_bits, bottom_label


def clustering(
    G, deg_corr=True, weighted=False, edge_distribution="best", np_nodes=None
):
    gtG = uti_gt.nx2gt(G, weight_type="double")
    communities, community_bits, bottom_label = clustering_from_gtG(
        gtG,
        deg_corr=deg_corr,
        weighted=weighted,
        edge_distribution=edge_distribution,
        np_nodes=np_nodes,
    )
    return communities, community_bits, bottom_label
