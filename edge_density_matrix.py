#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:26:04 2023

@author: kurodadaichi
"""

import numpy as np


# H matrix
# community_order is either list of numbers or dict keys
def calculate_H_matrix(nodes_list, bottom_communities, community_order=None):
    # bottom communities are list of lists or dict of lists
    # if dict of lists, keys are community names
    # if list of lists, community names are integers
    if community_order is None:
        if isinstance(bottom_communities, dict):
            community_order = sorted(bottom_communities.keys())
        else:
            community_order = range(len(bottom_communities))
    if not len(bottom_communities) == len(community_order):
        raise ValueError(
            "length of bottom_communities and community_order must be same"
        )
    H = np.array(
        [
            [1 * (_n in bottom_communities[c]) for c in community_order]
            for i, _n in enumerate(nodes_list)
        ],
        dtype=np.int8,
    )
    return H


# This function is taken from Schaub, Michael T., Jiaze Li, and Leto Peel. “Hierarchical Community Structure in Networks.” Physical Review E 107, no. 5 (May 23, 2023): 054305. https://doi.org/10.1103/PhysRevE.107.054305. and modified
def count_links_between_groups(
    A, H=None, bottom_communities=None, directed=False, self_loops=False, triu=False
):
    """
    Compute the number of possible and actual links between the groups
    indicated in the partition vector.
    NOTE: This function should only be used if A is the full adjacency
    matrix, otherwise possible_links will be calculated incorrectly.
    """
    if H is None and bottom_communities is None:
        raise ValueError("Either H or bottom_communities must be provided")
    elif H is None:
        # nodes_list = np.arange(A.shape[0], dtype=int)
        nodes_list = np.array(
            sorted(sum([list(bc) for bc in bottom_communities], [])), dtype=int
        )
        H = calculate_H_matrix(nodes_list, bottom_communities)

    # each block counts the number of half links / directed links
    links_between_groups = H.T @ A @ H
    # convert to dense matrix (if sparse, otherwise continue)
    try:
        links_between_groups = links_between_groups.A
    except AttributeError:
        pass

    # convert to array type first, before performing outer product
    nodes_per_group = np.ravel(H.sum(0))  # computing (d_{11},d_{22},...d_{kk})
    possible_links = np.outer(nodes_per_group, nodes_per_group)

    # if we do not allow self-loops this needs adjustment.
    if not self_loops:
        possible_links = possible_links - np.diag(nodes_per_group)

    if not directed:
        # we need to scale diagonal only by factor 2
        links_between_groups -= (np.diag(np.diag(links_between_groups)) / 2).astype(
            links_between_groups.dtype
        )
        if triu:
            links_between_groups = np.triu(links_between_groups)

        possible_links -= (np.diag(np.diag(possible_links)) / 2).astype(
            possible_links.dtype
        )
        if triu:
            possible_links = np.triu(possible_links)

    return links_between_groups, possible_links


def calc_p_bar(sp_A):
    return sp_A.sum() / (sp_A.shape[0] * (sp_A.shape[0] - 1))


def calc_edge_density_matrix(sp_A, bottom_communities):
    _num_edge_matrix, possible_counts = count_links_between_groups(
        sp_A, bottom_communities=bottom_communities
    )
    edge_density_matrix = _num_edge_matrix / possible_counts
    edge_density_matrix[(_num_edge_matrix == 0.0) * (possible_counts == 0.0)] = 0.0
    return edge_density_matrix
