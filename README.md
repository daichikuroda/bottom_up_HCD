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


- Paris codes (inside ./Paris/paris_codes/) are obtained from https://github.com/tbonald/paris.
- Beth-Hessian codes (files 'beth_hessian.py' and 'generic_functions.py') are obtained from https://lorenzodallamico.github.io/publication/unified20/.

:::
Note that graph tool sometimes conflicts with other libraries.
In our conda enviroment, graph tool and graphciz_layout cannot be used at the same time.
:::

## Getting Started
### aliases
- rbu stands for bottom-up
- rbp stands for top-down

### Trying on HSBM
You can try the codes on hierarchical stochastic block model with either trial_on_BTSBM.py or trial_on_unbalanced.py

#### k-narybalanced tree 
You can try by typing:
```
$ python3 trial_on_BTSBM.py beta a_last num_nodes_per_bottom_community number_of_levels k-nary
```

For example,
```
$ python3 trial_on_BTSBM.py 0.1 36 200 4 2
```
#### unbalanced tree example 1 or 2
You can try by typing:
```
$ python3 trial_on_unbalanced.py beta a_last example1/2 num_nodes_per_bottom_community
```

For example,
```
$ python3 trial_on_unbalanced.py 0.3 64 example1 100
```

### Trying on real datasets
#### high shool dataset
highschool.py

#### military dataset
trial_military_alliance.py

#### football dataset
football_nt.py