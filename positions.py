#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""
import numpy as np
import networkx as nx


def pos(G, num_layers, num_nodes, layout=nx.spring_layout):
    scale = 2 ** (-(num_layers))
    centers = [(0.0, 0.0)]
    for l in range(num_layers):
        if l % 2 == 0:
            centers = sorted(
                [(c[0] - 2 ** (-l // 2 - 1), c[1]) for c in centers]
                + [(c[0] + 2 ** (-l // 2 - 1), c[1]) for c in centers]
            )
        else:
            centers = sorted(
                [(c[0], c[1] - 2 ** (-l // 2 - 1)) for c in centers]
                + [(c[0], c[1] + 2 ** (-l // 2 - 1)) for c in centers]
            )
    pos = {
        k: v
        for ic in range(len(centers))
        for (k, v) in layout(
            list(range(ic * num_nodes, (ic + 1) * num_nodes)),
            center=centers[ic],
            scale=scale,
        ).items()
    }
    return pos


def pos_flex(G, clusterings, scale=None, layout=nx.spring_layout):
    if scale is None:
        scale = 1 / len(clusterings)
    # centers = [(0.0, 0.0)]
    centers = nx.circular_layout(range(len(clusterings)))
    # for c in centers.values():
    pos = {
        k: v
        for ic in range(len(centers))
        for (k, v) in layout(clusterings[ic], center=centers[ic], scale=scale).items()
    }
    return pos, centers


def pos_hierarchy_nchild(G, num_nodes, num_layers, nchild, layout=nx.spring_layout):
    scale = 1 / (nchild * (num_layers))
    centers = [(0.0, 0.0)]
    for l in range(num_layers):
        centers = sum(
            [
                list(
                    nx.circular_layout(
                        np.arange(3), center=c, scale=nchild ** (-l)
                    ).values()
                )
                for c in centers
            ],
            [],
        )
    pos = {
        k: v
        for ic in range(len(centers))
        for (k, v) in layout(
            list(range(ic * num_nodes, (ic + 1) * num_nodes)),
            center=centers[ic],
            scale=scale,
        ).items()
    }
    return pos


def pos_with_noise(G, num_layers, num_nodes, layout=nx.spring_layout, npseed=None):
    _rng = np.random.default_rng(npseed)
    scale = 2 ** (-(num_layers))
    centers = [(0.0, 0.0)]
    for l in range(num_layers):
        if l % 2 == 0:
            centers = sorted(
                [(c[0] - 2 ** (-l // 2 - 1), c[1]) for c in centers]
                + [(c[0] + 2 ** (-l // 2 - 1), c[1]) for c in centers]
            )
        else:
            centers = sorted(
                [(c[0], c[1] - 2 ** (-l // 2 - 1)) for c in centers]
                + [(c[0], c[1] + 2 ** (-l // 2 - 1)) for c in centers]
            )
    pos = {
        k: v
        for ic in range(len(centers))
        for (k, v) in layout(
            list(range(ic * num_nodes, (ic + 1) * num_nodes)),
            center=centers[ic] + 0.05 / _rng.random(2) * centers[ic][0],
            scale=scale,
        ).items()
    }
    return pos
