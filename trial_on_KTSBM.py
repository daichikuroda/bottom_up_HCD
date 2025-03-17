#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""
import numpy as np
import sys
import time
import csv
import utild
import HSBMs as hsbms
import wrapper as wra


def simulate(
    a_last,
    beta,
    num_nodes=200,
    num_layer=5,
    parent_seed=13,
    num_samples=10,
    num_child=2,
    stopping_rule="stop_bethe_hessian",
):

    num_true_communities = num_child ** (num_layer - 1)
    N = num_nodes * num_true_communities
    layers_to_see = list(range(1, num_layer))

    measurements = [
        "err_P",
        "err_St",
        "ami",
        "ari",
    ]

    algos = ["rbu", "rbp", "paris"]

    a_list = a_last * beta ** (np.arange(0, num_layer))[::-1]
    edge_densities = utild.calc_p_from_a(a_list, N)
    if edge_densities[-1] > 1.0:
        print(edge_densities)
        raise ValueError
    print("edge_densities: ", edge_densities)
    shsbm_model = hsbms.shsbm_nchild(num_nodes, num_child, edge_densities)

    G = shsbm_model.create_graph()
    com_detections = wra.community_detections(G, shsbm_model, algos)

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
                _algo,
                "number of estimated clusters",
                com_detections.algos[_algo].est_k(),
            )
        for _il, _layer in enumerate(layers_to_see):
            print(
                _algo,
                "accuracy on " + str(_layer),
                com_detections.calc_acc_on_l(_algo, _layer),
            )
    return com_detections


if __name__ == "__main__":
    arg = sys.argv
    beta = float(arg[1])
    a_last = int(arg[2])
    num_nodes = int(arg[3])
    num_layer = int(arg[4]) + 1
    num_child = int(arg[5])
    print("Simulation started at ", time.asctime(time.localtime()))
    com_detections = simulate(
        a_last,
        beta,
        num_nodes=num_nodes,
        num_layer=num_layer,
        num_child=num_child,
        stopping_rule="stop_bethe_hessian",
    )
    print("Simulation ended at ", time.asctime(time.localtime()))
