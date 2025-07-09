#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""
import numpy as np
import sys
import csv
import utils
import HSBMs as hsbms
import wrapper as wra

algos_labels = {
    "rbu": "bottom-up",
    "rbp": "top-down",
}


def simulate(
    a_list,
    num_nodes=200,
    num_layer=4,
    parent_seed=None,
    num_samples=10,
    num_child=2,
    rbp_with_sim=False,
):
    num_true_communities = num_child**num_layer
    N = num_nodes * num_true_communities
    layers_to_see = list(range(1, num_layer + 1))  # [2, 3, 4, 5]

    measurements = [
        "err_St",
        "ami",
    ]

    algos_main = ["rbu", "rbp"]
    algos = algos_main
    algos_k = [algo + "_k" for algo in algos_main]
    est_k_algos = algos_main

    acc_dict = {algo: np.zeros((len(layers_to_see), num_samples)) for algo in algos}
    ami_dict = {algo: np.zeros((len(layers_to_see), num_samples)) for algo in algos}

    measurement_dict = {
        _mea: {algo: np.zeros((num_samples)) for algo in algos + algos_k}
        for _mea in measurements
    }

    est_k_dict = {k: np.zeros((num_samples), dtype=int) for k in est_k_algos}

    _rng = np.random.default_rng(parent_seed)
    np_seeds = []
    edge_densities = utils.calc_p_from_a(a_list, N)
    if edge_densities[-1] > 1.0:
        print(edge_densities)
        raise ValueError
    print("edge_densities: ", edge_densities)
    shsbm_model = hsbms.shsbm_nchild(num_nodes, num_child, edge_densities)
    for iseed in range(num_samples):
        seed = _rng.integers(10**4)
        G = shsbm_model.create_graph(npseed=seed)
        np_seeds.append(seed)
        com_detections = wra.community_detections(G, shsbm_model, algos)

        for _algo in algos:
            for _mea in measurements:
                measurement_dict[_mea][_algo][iseed] = com_detections.calc_metric_err(
                    _algo, *wra.measurement_calc_funcs[_mea], maxk=False
                )
                measurement_dict[_mea][_algo + "_k"][iseed] = (
                    com_detections.calc_metric_err(
                        _algo, *wra.measurement_calc_funcs[_mea], maxk=True
                    )
                )
            est_k_dict[_algo][iseed] = com_detections.algos[_algo].est_k()
            for _il, _layer in enumerate(layers_to_see):
                print(_layer)
                acc_dict[_algo][_il][iseed] = com_detections.calc_acc_on_l(
                    _algo, _layer, rbp_with_sim=rbp_with_sim
                )
                ami_dict[_algo][_il][iseed] = com_detections.calc_ami_on_l(
                    _algo, _layer, rbp_with_sim=rbp_with_sim
                )

    for _mea in measurements:
        for _algo in algos + algos_k:
            print(_algo, _mea, "mean", np.mean(measurement_dict[_mea][_algo]))

    for _algo in est_k_algos:
        print(
            _algo,
            "mean estimated number of clusters",
            np.mean(est_k_dict[_algo]),
        )

    for _il, _layer in enumerate(layers_to_see):
        for _algo in algos:
            print(
                _algo,
                "accuracy on layer " + str(_layer),
                np.mean(acc_dict[_algo][_il]),
            )
            print(
                _algo,
                "AMI on layer " + str(_layer),
                np.mean(ami_dict[_algo][_il]),
            )
    return com_detections, ami_dict, acc_dict


if __name__ == "__main__":
    arg = sys.argv
    num_layer = int(arg[1])
    num_child = int(arg[2])
    num_nodes = int(arg[3])
    num_samples = int(arg[4])
    a_list = [float(a) for a in arg[5 : 6 + num_layer]]
    if len(arg) >= 8 + num_layer:
        if arg[7 + num_layer] != "None":
            try:
                parent_seed = int(arg[6 + num_layer])
            except:
                parent_seed = None
        else:
            parent_seed = None
    else:
        parent_seed = None
    rbp_with_sim = True
    com_dtections, amis, accs = simulate(
        a_list,
        num_nodes=num_nodes,
        num_layer=num_layer,
        parent_seed=parent_seed,
        num_child=num_child,
        num_samples=num_samples,
        rbp_with_sim=rbp_with_sim,
    )
