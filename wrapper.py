#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kurodadaichi
"""
import sys
import numpy as np
import networkx as nx
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score

import bethe_hessian as cla
import utild
import recursive as rec
import spectrald as spect
import measurements as mea
import plots
import importlib

spam_spec = importlib.util.find_spec("graph_tool")
gt_found = spam_spec is not None
if gt_found:
    import bayesian_clsutering as bc
else:
    print("cannot find graph tool.")
    print("You cannot try bayesian.")
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

# nodes are expected to be labeled as 0,1,2,...,N
# in Z the cluster number is consecutive number starting from 0
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
        deg_corr=True,
        edge_distribution="best",
        nodes_sort=True,
    ):
        if not algo in [
            "rbu",
            "rbp",
            "top-down",
            "top_down",
            "bottom-up",
            "bottom_up",
            "paris",
        ]:
            if algo != "bayesian" or not gt_found:
                print("invalid algo")
                algo = ""
        self.N = len(G.nodes())
        self.algo = algo
        if nodes_sort:
            self.nodes_list = np.array(sorted(G.nodes()))
        else:
            self.nodes_list = np.array(G.nodes())
        self.np_A = nx.to_numpy_array(G, nodelist=self.nodes_list)
        if group_name is None:
            self.true_label = None
            self.groups = None
        else:
            self.true_label = [
                v for (_k, v) in sorted(dict(G.nodes(data=group_name)).items())
            ]
            self.groups = {}
            for n, g in G.nodes(data="value"):
                _l = self.groups.get(g, [])
                _l.append(n)
                self.groups[g] = _l
        if num_communities is not None:
            self.num_communities = num_communities
        if metrics is None:
            metrics = metrics_set
        else:
            if "St" in metrics:
                metrics.append("small_St")
            if "Stk" in metrics:
                metrics.append("small_Stk")
            metrics = list(set(metrics) & metrics_set)
        self.metrics = {met: None for met in metrics}
        self.d_mapping_for_paris = None
        if algo == "rbu" or algo == "bottom_up" or algo == "bottom-up":
            if initial_communities is None and initial_labels is None:
                self.bottom_label, _zeta_p = cla.community_detection(
                    G, weighted, increase_maxiter=100, nodelist=self.nodes_list
                )
                self.bottom_communities = utild.return_communities(
                    self.bottom_label, np_nodes=self.nodes_list
                )
            elif initial_labels is not None:
                self.bottom_communities = utild.return_communities(
                    initial_labels, np_nodes=self.nodes_list
                )
                self.bottom_label = initial_labels
            elif initial_communities is not None:
                self.bottom_communities = initial_communities
                self.bottom_label = utild.communities_to_label(
                    self.bottom_communities, nodes_list=self.nodes_list
                )
            self.Z = rec.bottom_up(
                G,
                self.bottom_label,
                linkage_algo="update_each",
                weighted=weighted,
                nodelist=self.nodes_list,
            )
            self.label = self.bottom_label
            self.communities = self.bottom_communities
        elif algo == "rbp" or algo == "top_down" or algo == "top-down":
            if initial_labels is not None:
                initial_communities = utild.return_communities(
                    initial_labels, np_nodes=self.nodes_list
                )
            (self.bottom_communities, self.community_bits,) = rec.recursive_bipartion(
                G,
                partion_algo=spect.regularized_spectral,
                stopping_rule=spect.stop_bethe_hessian,
                initial_communities=initial_communities,
            )
            self.Z = spect.linkage_for_recursive_algo(self.community_bits)
            self.bottom_label = utild.communities_to_label(
                self.bottom_communities, nodes_list=self.nodes_list
            )
            self.similarities = mea.calc_similarities_for_top_down(
                self.np_A, self.Z, self.bottom_communities, nodes_list=self.nodes_list
            )
            self.label = self.bottom_label
            self.communities = self.bottom_communities
        elif algo == "paris":
            # in bayesian the node number should be consecutive from 0
            graph_num_convert = (
                self.nodes_list[0] != 0
                or self.nodes_list[-1] != len(self.nodes_list) - 1
            )
            if graph_num_convert:
                self.e_mapping_for_paris = dict(
                    zip(self.nodes_list, range(len(self.nodes_list)))
                )
                self.d_mapping_for_paris = dict(
                    zip(range(len(self.nodes_list)), self.nodes_list)
                )
                G2 = nx.relabel_nodes(G, self.e_mapping_for_paris)
            else:
                G2 = G
            self.Z = paris(G2)
            self.bottom_communities = self.nodes_list.reshape((self.N, 1))
            self.communities = utils.best_clustering(self.Z, 0)
            print(len(self.communities))
            if graph_num_convert:
                self.communities = [
                    np.array([self.d_mapping_for_paris[n] for n in c])
                    for c in self.communities
                ]

            self.label = utild.communities_to_label(
                self.communities, nodes_list=self.nodes_list
            )

        elif algo == "bayesian":
            # in bayesian the node number should be consecutive from 0
            graph_num_convert = (
                self.nodes_list[0] != 0
                or self.nodes_list[-1] != len(self.nodes_list) - 1
            )
            if graph_num_convert:
                e_mapping_for_paris = dict(
                    zip(self.nodes_list, range(len(self.nodes_list)))
                )
                d_mapping_for_paris = dict(
                    zip(range(len(self.nodes_list)), self.nodes_list)
                )
                G2 = nx.relabel_nodes(G, e_mapping_for_paris)
            else:
                G2 = G
            (
                self.bottom_communities,
                self.community_bits,
                self.bottom_label,
            ) = bc.clustering(
                G2,
                deg_corr=deg_corr,
                weighted=weighted,
                np_nodes=np.array(self.nodes_list),
                edge_distribution=edge_distribution,
            )
            self.Z = spect.linkage_for_recursive_algo(self.community_bits)
            if graph_num_convert:
                self.bottom_communities = [
                    np.array([d_mapping_for_paris[n] for n in c])
                    for c in self.bottom_communities
                ]
            self.label = self.bottom_label
            self.communities = self.bottom_communities
        if num_communities is not None:
            self.communitiesk = utild.clustering_k_communities(
                self.Z, self.num_communities, self.bottom_communities
            )
            self.labelk = utild.communities_to_label(
                self.communitiesk, nodes_list=self.nodes_list
            )

    def est_k(self):
        return len(self.communities)

    def estimate_metric(self, metric):
        if metric[-1] == "k":
            maxk = self.num_communities
        else:
            maxk = self.est_k()
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

    def convert_communities_of_paris(self, input_communities):
        if self.d_mapping_for_paris is None:
            return input_communities
        else:
            return [
                np.array([self.d_mapping_for_paris[n] for n in c])
                for c in input_communities
            ]


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
        if (
            "ttbtt" in algos
            and len(set(["rbp", "top-down", "top_down"]) & set(algos)) == 0
        ):
            algos.append("rbp")
            is_ttbtt = True
        elif "ttbtt" in algos:
            is_ttbtt = True
        else:
            is_ttbtt = False
        algos2 = list(set(algos) - set(["ttbtt"]))
        self.algos = {
            _algo: hierarchical_communities(
                self.G,
                _algo,
                self.hsbm_model.num_communities,
                metrics=metrics,
            )
            for _algo in algos2
        }
        if is_ttbtt:
            self.algos["ttbtt"] = hierarchical_communities(
                self.G,
                "rbu",
                self.hsbm_model.num_communities,
                metrics=metrics,
                initial_communities=self.algos["rbp"].bottom_communities,
            )
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

    def calc_acc_on_l(self, algo, layer, rbp_with_sim=False):
        if algo in ["rbp", "top-down", "top_down"] and rbp_with_sim:
            clustering = utild.clustering_k_communities_by_similarities(
                self.algos[algo].Z,
                self.hsbm_model.num_clusters_on_l(layer),
                self.algos[algo].bottom_communities,
                self.algos[algo].similarities,
            )
        else:
            clustering = utild.clustering_k_communities(
                self.algos[algo].Z,
                self.hsbm_model.num_clusters_on_l(layer),
                self.algos[algo].bottom_communities,
            )
        return self.hsbm_model.calc_accuracy(
            clustering, layer, calc_acc_algo=mea.calc_accuracy
        )

    def calc_ami_on_l(self, algo, layer, rbp_with_sim=False):
        if algo in ["rbp", "top-down", "top_down"] and rbp_with_sim:
            clustering = utild.clustering_k_communities_by_similarities(
                self.algos[algo].Z,
                self.hsbm_model.num_clusters_on_l(layer),
                self.algos[algo].bottom_communities,
                self.algos[algo].similarities,
            )
        else:
            clustering = utild.clustering_k_communities(
                self.algos[algo].Z,
                self.hsbm_model.num_clusters_on_l(layer),
                self.algos[algo].bottom_communities,
            )
        label_on_l = utild.communities_to_label(
            clustering, nodes_list=self.algos[algo].nodes_list
        )
        similarity = layer
        true_clustering_on_l = self.hsbm_model.true_clustering(similarity)
        true_label_on_l = utild.communities_to_label(
            true_clustering_on_l, nodes_list=self.hsbm_model.nodelist
        )
        return adjusted_mutual_info_score(true_label_on_l, label_on_l)
