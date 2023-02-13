#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:48:15 2022

@author: kurodadaichi
"""

import urllib.request
import io
import zipfile
import networkx as nx
import numpy as np
import utild
import csv
import positions


def normal(x):
    return x


def normalize(x):
    return (x - np.mean(x)) / np.std(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize_sigmoid(x):
    nx = normalize(x)
    return sigmoid(nx)


def treat_weight(G, data_name, method=normal):
    edges_with_weights = G.edges(data=data_name)
    weights = np.array([w for (_e0, _e1, w) in edges_with_weights])
    processed = method(weights)
    return [(e0, e1, pw) for (pw, (e0, e1, _w)) in zip(processed, edges_with_weights)]


def create_prime_school_net(
    gexf_file, data_name="duration", to_int_label=False, weight_treat=normal
):
    G = nx.read_gexf(gexf_file, version="1.1draft", relabel=True)
    if to_int_label:
        mapping = dict(zip(G.nodes(), [int(n) for n in G.nodes()]))
        G = nx.relabel_nodes(G, mapping)
    G.add_weighted_edges_from(treat_weight(G, "duration", method=weight_treat))
    nodes_classname = dict(G.nodes(data="classname"))
    class_names = set(nodes_classname.values())
    classes = {k: [] for k in class_names}
    classes = {k: v for (k, v) in sorted(classes.items())}
    for k, v in nodes_classname.items():
        classes[v].append(k)
    pos = positions.pos_flex(G, list(classes.values()))
    edges = G.edges()
    weights = np.array([G[u][v]["weight"] for u, v in edges])
    return G, classes, pos, weights


def network_from_edgelist(edge_list, weighted=True):
    G = nx.Graph()
    for i, j, w in edge_list:
        if weighted:
            G.add_edge(int(i), int(j), weight=w)
        else:
            G.add_edge(int(i), int(j))
    return G


def change_node_nums(G):
    mapping = dict(zip(G.nodes(), range(len(G.nodes()))))
    to_get_original_mapping = dict(zip(range(len(G.nodes())), G.nodes()))
    G = nx.relabel_nodes(G, mapping)
    return G, to_get_original_mapping, mapping


def high_school_net():
    high_school_folder = "./high_school_data/"
    # contact_diaries_net_f = high_school_folder + "Contact-diaries-network_data_2013.csv"
    contact_net_f = high_school_folder + "High-School_data_2013.csv"
    datas = np.loadtxt(contact_net_f, dtype=int, usecols=(1, 2))
    meta_data_f = high_school_folder + "metadata_2013.txt"
    edges, weights = np.unique(datas, axis=0, return_counts=True)
    weights = np.log(weights)
    edge_list = np.hstack((edges, weights.reshape((len(weights), 1))))
    G = network_from_edgelist(edge_list)
    # G, to_get_original_mapping, mapping = change_node_nums(G)
    class_dict = {}
    gender_dict = {}
    with open(meta_data_f, "r") as f:
        csvreader = csv.reader(f, delimiter="\t")
        for row in csvreader:
            class_dict[int(row[0])] = row[1]
            gender_dict[int(row[0])] = row[2]

    nx.set_node_attributes(G, class_dict, "class")
    nx.set_node_attributes(G, gender_dict, "gender")
    return G  # , to_get_original_mapping, mapping


def handle_inf_in_Z(Z, method="multiply", quantity=2):
    iinf = Z.T[2] == np.inf
    max_e = np.max(Z.T[2][~iinf])
    if method == "multiply":
        Z.T[2][iinf] = max_e * quantity
    elif method == "add":
        Z.T[2][iinf] = max_e + quantity
    return Z


def shuffle_node_nums(G):
    nums = list(G.nodes())
    shuffled = nums.copy()
    np.random.shuffle(nums)
    encode_mapping = dict(zip(nums, shuffled))
    decode_mapping = dict(zip(shuffled, nums))
    G = nx.relabel_nodes(G, encode_mapping)
    return G, encode_mapping, decode_mapping


def create_football_net():
    url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

    sock = urllib.request.urlopen(url)  # open URL
    s = io.BytesIO(sock.read())  # read into BytesIO "file"
    sock.close()

    zf = zipfile.ZipFile(s)  # zipfile object
    txt = zf.read("football.txt").decode()  # read info file
    gml = zf.read("football.gml").decode()  # read gml data
    # throw away bogus first line with # from mejn files
    gml = gml.split("\n")[1:]
    G = nx.parse_gml(gml)  # parse gml data
    return G, txt


def data_correction(G):
    correction_dict0 = {
        11: (12, 10, "N.Texas"),
        24: (12, 10, "Arkansas State"),
        28: (12, 11, "Boise State"),
        50: (12, 10, "Idaho"),
        58: (5, 11, "Louisiana Tech"),
        59: (5, 10, "Louisiana Monroe"),
        63: (5, 10, "Middle Tennessee State"),
        69: (12, 10, "New Mexico State"),
        90: (12, 5, "Utah State"),
        97: (5, 10, "Louisiana Lafayette"),
        110: (11, 4, "Texas Christian"),
    }
    # correction_dict = {
    #     "N.Texas": (12, 10, 11),
    #     "Arkansas State": (12, 10, 24),
    #     "Boise State": (12, 11, 28),
    #     "Idaho": (12, 10, 50),
    #     "Louisiana Tech": (5, 11, 58),
    #     "Louisiana Monroe": (5, 10, 59),
    #     "Middle Tennessee State": (5, 10, 63),
    #     "New Mexico State": (12, 10, 69),
    #     "Utah State": (12, 5, 90),
    #     "Louisiana Lafayette": (5, 10, 97),
    #     "Texas Christian": (11, 4, 110),
    # }
    for n, v in G.nodes("value"):
        to_correct = correction_dict0.get(n, v)
        if type(to_correct) is tuple:
            if to_correct[1] == v:
                G.nodes[n]["value"] = to_correct[0]
            else:
                print("error!!!", G.nodes[n]["value"], to_correct)
    return G
