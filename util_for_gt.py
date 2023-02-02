#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""

import networkx as nx
import graph_tool.all as gt
import numpy as np

# from scipy.cluster.vq import kmeans2


def to_graph_tool(A, directed=False, dtype="int"):
    g = gt.Graph(directed=directed)
    edge_weights = g.new_edge_property(dtype)
    # edge_weights = g.new_edge_property('double')
    g.edge_properties["weight"] = edge_weights
    nnz = np.nonzero(np.triu(A, 1))
    nedges = len(nnz[0])
    g.add_edge_list(
        np.hstack([np.transpose(nnz), np.reshape(A[nnz], (nedges, 1))]),
        eprops=[edge_weights],
    )
    return g


def nx_to_gt(nxG):
    A = nx.to_numpy_array(nxG)
    gtG = to_graph_tool(A, directed=nx.is_directed(nxG))
    return gtG


# there is also code here
# https://bbengfort.github.io/2016/06/graph-tool-from-networkx/
def nx2gt(nxG, weight="weight", weight_type="int"):
    g = gt.Graph(directed=nx.is_directed(nxG))
    edge_list = nx.to_edgelist(nxG)
    edge_list = [tuple(list(edge[:2]) + [edge[2].get(weight, 1)]) for edge in edge_list]
    # print(edge_list)
    edge_weights = g.new_edge_property(weight_type)
    # print(edge_weights)
    # for _n in G.nodes():
    #     g.add_vertex(_n)
    g.edge_properties[weight] = edge_weights
    g.add_edge_list(edge_list, eprops=[edge_weights])
    nodes_to_add = max(nxG.nodes()) - max(np.unique([[k,v] for k, v, _w in edge_list]))
    g.add_vertex(nodes_to_add)
    return g


def infer_partion(nxG, algo=gt.minimize_blockmodel_dl, weighted=False, deg_corr=False):
    if weighted:
        gtG = nx_to_gt(nxG, dtype="double")
    else:
        gtG = nx_to_gt(nxG)
    state = algo(gtG, state_args=dict(deg_corr=deg_corr))
    label = state.get_state()
    if type(state) is list:
        label = label[0]
    label = np.array(list(label))
    # TO_DO: check if the entropy is consistent for deg_corr = True & False
    # entropy = ""
    return label  # , entropy

