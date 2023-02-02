#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""
import sys
import numpy as np
import networkx as nx
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score

import beth_hessian as cla
import utild
import recursive as rec
import spectrald as spect
import measurements as mea
import plots
import bayesian_clsutering as bc

sys.path += ["./Paris/paris_codes/"]
from paris import paris
import utils

measurement_calc_funcs = {
    "err_P": ("P", "normal"),
    "err_St": ("St", "normal"),
    "ami": ("label", "ami"),
    "ari": ("label", "ari"),
    "err_St_old": ("St", "old"),
    "l21_St_err": ("St", "l21"),
    "ave_St_err": ("St", "simple_ave"),
    "ave_P_err": ("P", "simple_ave"),
}

St_rob_calc_funcs = {
    "err_St_hat": ("hat_St", mea.calc_standrized_square_error_from_matrixes),
    "err_St_hat_old": ("hat_St", mea.calc_old_err_St),
    "average_St_hat": ("hat_St", utild.simple_ave),
}

metrics_set = {"St", "Stk", "small_St", "small_Stk", "P", "Pk"}


calc_funcs = {
    "normal": mea.calc_standrized_square_error_from_matrixes,
    "old": mea.calc_old_err_St,
    "simple_ave": utild.simple_ave,
    "l21": utild.l21,
    "ami": adjusted_mutual_info_score,
    "ari": adjusted_rand_score,
}


class hierarchical_communities:
    def __init__(
        self,
        G,
        algo,
        num_communities=None,
        metrics=None,
        initial_communities=None,
        initial_labels=None,
        weighted=False,
        group_name=None,  # "values",
        continuous_node_label=True,
        deg_corr=True,
        edge_distribution="best",
    ):
        self.N = len(G.nodes())
        self.algo = algo
        self.np_A = nx.to_numpy_array(G)
        if group_name is None:
            self.true_label = None
            self.groups = None
        else:
            self.true_label = [v for v in dict(G.nodes(data=group_name)).values()]
            self.groups = {}
            for n, g in G.nodes(data="value"):
                _l = self.groups.get(g, [])
                _l.append(n)
                self.groups[g] = _l
        if num_communities is not None:
            self.num_communities = num_communities
        if continuous_node_label is False:
            nodes_list = np.array(G.nodes())
        else:
            nodes_list = None
        if metrics is None:
            metrics = metrics_set
        else:
            if "St" in metrics:
                metrics.append("small_St")
            if "Stk" in metrics:
                metrics.append("small_Stk")
            metrics = list(set(metrics) & metrics_set)
        self.metrics = {met: None for met in metrics}
        if algo == "rbu" or algo == "bottom_up" or algo == "bottom-up":
            if initial_communities is None and initial_labels is None:
                self.bottom_label, _zeta_p = cla.community_detection(
                    G, weighted, increase_maxiter=100
                )
                self.bottom_communities = utild.return_communities(
                    self.bottom_label, np_nodes=nodes_list
                )
            elif initial_labels is not None:
                self.bottom_communities = utild.return_communities(
                    initial_labels, np_nodes=nodes_list
                )
                self.bottom_label = initial_labels
            elif initial_communities is not None:
                self.bottom_communities = initial_communities
                self.bottom_label = utild.communities_to_label(
                    self.bottom_communities, nodes_list=nodes_list
                )
            self.Z = rec.bottom_up(
                G, self.bottom_label, linkage_algo="update_each", weighted=weighted
            )
            self.label = self.bottom_label
            self.communities = self.bottom_communities
        elif algo == "rbp" or algo == "top_down" or algo == "top-down":
            if initial_labels is not None:
                initial_communities = utild.return_communities(
                    initial_labels, np_nodes=nodes_list
                )
            (self.bottom_communities, self.community_bits,) = rec.recursive_bipartion(
                G,
                partion_algo=spect.regularized_spectral,
                stopping_rule=spect.stop_bethe_hessian,
                initial_communities=initial_communities,
            )
            self.Z = spect.linkage_for_recursive_algo(self.community_bits)
            self.bottom_label = utild.communities_to_label(
                self.bottom_communities, nodes_list=nodes_list
            )
            self.label = self.bottom_label
            self.communities = self.bottom_communities
        elif algo == "paris":
            self.Z = paris(G)
            self.bottom_communities = np.arange(self.N).reshape((self.N, 1))
            self.communities = utils.best_clustering(self.Z, 0)
            self.label = utild.communities_to_label(
                self.communities, nodes_list=nodes_list
            )
        elif algo == "bayesian":
            (
                self.bottom_communities,
                self.community_bits,
                self.bottom_label,
            ) = bc.clustering(
                G,
                deg_corr=deg_corr,
                weighted=weighted,
                np_nodes=np.array(nodes_list),
                edge_distribution=edge_distribution,
            )
            self.Z = spect.linkage_for_recursive_algo(self.community_bits)
            self.label = self.bottom_label
            self.communities = self.bottom_communities
        if num_communities is not None:
            self.communitiesk = utild.clustering_k_communities(
                self.Z, self.num_communities, self.bottom_communities
            )
            self.labelk = utild.communities_to_label(
                self.communitiesk, nodes_list=nodes_list
            )

    def est_k(self):
        return len(self.communities)

    def estimate_metric(self, metric):
        if metric[-1] == "k":
            maxk = self.num_communities
        else:
            maxk = self.est_k()
        # met = self.metrics[metric]
        # if met is None and metric in ["P", "Pk"]:
        #     met = mea.estimate_p_matrix(
        #         self.np_A,
        #         self.Z,
        #         self.bottom_communities,
        #         maxk=maxk,
        #     )
        # elif met is None and metric in ["St", "Stk", "small_St", "small_Stk"]:
        #     St, St_small = mea.tree_similarity_matrix(
        #         self.Z,
        #         self.bottom_communities,
        #         maxk=maxk,
        #     )
        #     if metric[:4] == "small":
        #         self.metrics[metric] = St_small
        #         self.metrics[metric[4:]] = St
        #         met = St_small
        #     else:
        #         self.metrics[metric] = St
        #         self.metrics["small_" + metric] = St_small
        #         met = St
        if self.metrics[metric] is None:
            St, St_small, P = mea.Sts_P(
                self.np_A,
                self.Z,
                self.bottom_communities,
                maxk=maxk,
            )
            if metric[-1] == "k":
                self.metrics["Stk"] = St
                self.metrics["small_Stk"] = St_small
                self.metrics["Pk"] = P
            else:
                self.metrics["St"] = St
                self.metrics["small_St"] = St_small
                self.metrics["P"] = P
        return self.metrics[metric]

    def draw_network_and_dendrogram(
        self,
        G,
        weights=1,
        save_path=None,
        original_edges=False,
        node_size=3,
        edges_alpha=0.3,
        knot_size=4,
        legend_on=True,
        legend_size=None,
    ):
        dG, _c = utild.nx_dendrogram(self.Z, self.bottom_communities)
        plots.draw_network_and_dendrogram(
            G,
            dG,
            self.groups,
            weights=1,
            save_path=None,
            original_edges=False,
            node_size=3,
            edges_alpha=0.3,
            knot_size=4,
            legend_on=True,
            legend_size=None,
        )


class community_detections:
    def __init__(
        self,
        G,
        hsbm_model,
        algos,
        metrics=None,
        deg_corr=False,
    ):
        self.G = G
        self.hsbm_model = hsbm_model
        self.St_true = self.hsbm_model.true_St()
        self.true_label = self.hsbm_model.true_label
        self.algos = {
            _algo: hierarchical_communities(
                self.G,
                _algo,
                self.hsbm_model.num_communities,
                metrics=metrics,
                deg_corr=deg_corr,
            )
            for _algo in algos
        }
        self.true_metrics = {
            "P": self.hsbm_model.probability_matrix,
            "St": self.hsbm_model.true_St(),
            "small_St": self.hsbm_model.true_St_small(),
        }

    def calc_metric_err(self, algo, metric, calc, maxk=False):
        calc = calc_funcs[calc]
        if metric != "label":
            _metric = metric
            if maxk is True:
                _metric += "k"
            estimated_metric = self.algos[algo].estimate_metric(_metric)
            return calc(estimated_metric, self.true_metrics[metric])
        else:
            if maxk is True:
                detected_label = self.algos[algo].labelk
            elif maxk is False:
                detected_label = self.algos[algo].label
            return calc(self.hsbm_model.true_label, detected_label)

    def calc_acc_on_l(self, algo, layer):
        clustering = utild.clustering_k_communities(
            self.algos[algo].Z,
            self.hsbm_model.num_clusters_on_l(layer),
            self.algos[algo].bottom_communities,
        )
        return self.hsbm_model.calc_accuracy(
            clustering, layer, calc_acc_algo=mea.calc_accuracy
        )
