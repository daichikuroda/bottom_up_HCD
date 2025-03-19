# bottom-up hierarchical community detection algorithm

## Dependency
This codes depends on follwing codes:
- networkx
    - graphviz_layout
- scipy
- sklearn
- numpy
- matplotlib

- graph tool (bayesian)
- geopandas (plotting military dataset)


- Paris codes (inside ./Paris/paris_codes/) are obtained from https://github.com/tbonald/paris. This method is introduced in [Hierarchical Graph Clustering using Node Pair Sampling](http://arxiv.org/abs/1806.01664).
- Beth-Hessian codes (files 'beth_hessian.py' and 'generic_functions.py') are obtained from https://lorenzodallamico.github.io/publication/unified20/. This method is introduced in [Hierarchical block structures and high-resolution model selection in large networks](https://doi.org/10.1103/PhysRevX.4.011047).
:::
Note that graph tool sometimes conflicts with other libraries.
In our conda enviroment, graph tool and graphciz_layout cannot be used at the same time.
:::
- Power grid of Continental Europe from the [Union for the Coordination of Transmission of Electricity (UCTE) map](http://www.ucte.org) is obtained from the [github repository](https://github.com/barahona-research-group/PyGenStability/blob/master/examples/real_examples/powergrid/Example_powergrid.ipynb), which is used for [Markov Dynamics as a Zooming Lens for Multiscale Community Detection: Non Clique-Like Communities and the Field-of-View Limit](https://doi.org/10.1371/journal.pone.0032210).

## Getting Started
### Aliases
- rbu stands for bottom-up
- rbp stands for top-down

### Trying on HSBM
You can try the codes on hierarchical stochastic block model with either trial_on_KTSBM.py or trial_on_unbalanced.py

#### K-nary balanced tree 
You can try by typing:
```
$ python3 trial_on_BTSBM.py beta a_last num_nodes_per_bottom_community number_of_levels k-nary
```

For example,
```
$ python3 trial_on_BTSBM.py 0.1 36 200 4 2
```
#### Unbalanced tree example 1 or 2
You can try by typing:
```
$ python3 trial_on_unbalanced.py beta a_last example1/2 num_nodes_per_bottom_community
```

For example,
```
$ python3 trial_on_unbalanced.py 0.3 64 example1 100
```

### Robustness of linkage to misculstering errors
#### Type of errors and tree recovery rate without graph split
robustness_bu_bh.py

#### Tree recovery rate with graph split
robustness_bu_bh_graph_split.py

#### Confusion matrix
robustness_bu_bh_plot.py

### Trying on real datasets
#### High shool dataset
highschool.py

#### Military dataset
trial_military_alliance.py

#### Football dataset
football_nt.py