#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""
import numpy as np
import sys
import time
import os
import csv
import utild
import HSBMs as hsbms
import wrapper as wra

parent_codef = "/home/indy-stg3/user2/codes/"

sizes_merge = {
    "example1": [2, 2] + [1] * 24 + [2, 2],
    "example2": [2, 2] + [1] * 12 + [8, 8],
}


def simulate(
    a_last,
    beta,
    tree_shape="example1",
    num_nodes_unit=100,
    tree_type="merge_original",
    num_layer=6,
    stopping_rule="stop_bethe_hessian",
):
    f = tree_shape + "_group_matrix.csv"
    if not os.path.exists(f):
        f = parent_codef + tree_shape + "_group_matrix.csv"
    group_matrix = np.loadtxt(f, delimiter=",", dtype=int)
    num_true_communities = len(group_matrix)
    if (
        tree_type == "merge"
        or tree_type == "merge_original"
        or tree_type == "merge_original2"
    ):
        sizes = np.array(sizes_merge[tree_shape]) * num_nodes_unit
    elif tree_type == "equal":
        sizes = np.ones(num_true_communities, dtype=int) * num_nodes_unit
    N = sum(sizes)

    layers_to_see = [1, 2]

    measurements = [
        "err_P",
        "err_St",
        "ami",
        "ari",
    ]
    algos = ["rbu", "rbp"]

    a_list = a_last * beta ** (np.arange(0, num_layer))[::-1]
    edge_densities = utild.calc_p_from_a(a_list, N)
    if edge_densities[-1] > 1.0:
        print(edge_densities)
        raise ValueError
    print("edge_densities: ", edge_densities)
    if tree_type == "merge_original":
        _p0_n = (edge_densities[::-1][0] + edge_densities[::-1][1]) / 2
        _p0_n1 = (
            edge_densities[::-1][0]
            + edge_densities[::-1][1]
            + 2 * edge_densities[::-1][2]
            + 4 * +edge_densities[::-1][3]
        ) / 8
        specified_ps_dict = {
            "example1": [
                (0, 0, _p0_n),
                (1, 1, _p0_n),
                (26, 26, _p0_n),
                (27, 27, _p0_n),
            ],
            "example2": [
                (0, 0, _p0_n),
                (1, 1, _p0_n),
                (14, 14, _p0_n1),
                (15, 15, _p0_n1),
            ],
        }
        hsbm_model = hsbms.hsbm(
            sizes,
            group_matrix,
            edge_densities,
            specified_ps=specified_ps_dict[tree_shape],
        )
    elif tree_type == "merge_original2":
        specified_ps_dict = {
            "example1": [
                (0, 0, edge_densities[::-1][1]),
                (1, 1, edge_densities[::-1][1]),
                (26, 26, edge_densities[::-1][1]),
                (27, 27, edge_densities[::-1][1]),
            ],
            "example2": [
                (0, 0, edge_densities[::-1][1]),
                (1, 1, edge_densities[::-1][1]),
                (14, 14, edge_densities[::-1][3]),
                (15, 15, edge_densities[::-1][3]),
            ],
        }
        hsbm_model = hsbms.hsbm(
            sizes,
            group_matrix,
            edge_densities,
            specified_ps=specified_ps_dict[tree_shape],
        )
    else:
        hsbm_model = hsbms.hsbm(sizes, group_matrix, edge_densities)

    G = hsbm_model.create_graph()
    com_detections = wra.community_detections(G, hsbm_model, algos)

    for _algo in algos:
        for _mea in measurements:
            print(
                _algo,
                _mea,
                com_detections.calc_metric_err(
                    _algo, *wra.measurement_calc_funcs[_mea], maxk=False
                ),
            )
            print(
                _algo,
                _mea,
                "number of clusters is given",
                com_detections.calc_metric_err(
                    _algo, *wra.measurement_calc_funcs[_mea], maxk=True
                ),
            )
        print(
            _algo, "number of estimated clusters", com_detections.algos[_algo].est_k()
        )
        for _il, _layer in enumerate(layers_to_see):
            print(
                _algo,
                "accuracy on " + str(_layer),
                com_detections.calc_acc_on_l(_algo, _layer),
            )
    return


if __name__ == "__main__":
    arg = sys.argv
    # arg = ["", 0.2, 30, "example1", 100]
    beta = float(arg[1])
    a_last = float(arg[2])
    tree_shape = arg[3]
    num_nodes_unit = int(arg[4])
    print("Simulation started at ", time.asctime(time.localtime()))
    simulate(
        a_last,
        beta,
        tree_shape,
        num_nodes_unit,
        # tree_type = tree_type,
        tree_type="merge",
        num_layer=6,
        stopping_rule="stop_bethe_hessian",
    )
    print("Simulation ended at ", time.asctime(time.localtime()))
