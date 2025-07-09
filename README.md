# Bottom-up hierarchical community detection algorithm

## Dependency
This codes depends on follwing libraries: 
```
python                    3.12.2 
numpy                     1.26.4
scikit-learn              1.5.1
pandas                    2.2.1
scipy                     1.13.1
seaborn                   0.13.2 
networkx                  3.1 
matplotlib                3.8.0 
tqdm                      4.67.1
igraph                    0.11.4
pygraphviz                1.9
geopandas                 0.14.2
scikit-bio                0.6.3
urllib3                   2.2.2
```

- Beth-Hessian codes (files 'beth_hessian.py') are obtained from https://lorenzodallamico.github.io/publication/unified20/. This method is introduced in [Dall'Amico, Lorenzo, Romain Couillet, and Nicolas Tremblay. "A unified framework for spectral clustering in sparse graphs." Journal of Machine Learning Research 22.217 (2021): 1-56.](https://www.jmlr.org/papers/v22/20-261.html).
- High school data is from [SocioPatterns](http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/). Originally studied in [Mastrandrea, Rossana, Julie Fournet, and Alain Barrat. "Contact patterns in a high school: a comparison between data collected using wearable sensors, contact diaries and friendship surveys." PloS one 10.9 (2015): e0136497.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136497)
- Power grid of Continental Europe from the [Union for the Coordination of Transmission of Electricity (UCTE) map](http://www.ucte.org) is obtained from the [github repository](https://github.com/barahona-research-group/PyGenStability/blob/master/examples/real_examples/powergrid/Example_powergrid.ipynb), which is used for [Schaub, Michael T., et al. "Markov dynamics as a zooming lens for multiscale community detection: non clique-like communities and the field-of-view limit." PloS one 7.2 (2012): e32210.](https://doi.org/10.1371/journal.pone.0032210).
- Military Alliance is taken from [ATOP project](http://www.atopdata.org/data.html) (direct download link: http://www.atopdata.org/uploads/6/9/1/3/69134503/atop_5.1__.dta_.zip)
- Foot ball data is from [M.E.J. Newman's webpage](http://www-personal.umich.edu/~mejn/netdata). (direct download link: http://www-personal.umich.edu/~mejn/netdata/football.zip)

## On HSBMs

### On BTSBMs (Figure 2) and ternary tree SBMs (Figure 8)
To reproduce the results of Figure 2, you can run the following command by varying the following parameters:
- number_of_levels: height of the tree
- k-nary: set k=2 for binary hierarchies and k=3 for ternary hierarchies
- num_nodes_per_bottom_community
- num_samples: The results will be aggregated over "num_samples" runs
- a_0, a_1, ..., a_{number_of_levels}: edge connection probabilities for level $\ell$ is set to $a_\ell \log N / N$, with $N$ being the number of nodes in the graph
- (seed): addding a seed (optional)
```
$ python3 KTSBM_plane.py number_of_levels k-nary num_nodes_per_bottom_community num_samples a_0 a_1 ... a_{number_of_levels} (seed)
```

For example, to obtain one-sample values at a_1 = 50, a_2 = 60 for Figure 2, 
```
$ python3 KTSBM_plane.py 3 2 200 1 40 50 60 100
``` 

To obtain one-sample values at a_1 = 50, a_2 = 60 for Figure 8, 
```
$ python3 KTSBM_plane.py 3 3 100 1 10 50 60 130
``` 


### Unbalanced tree example 1 (Figure 9) and example 2 (Figure 10)
To reproduce the results of Figure 2, you can run the following command by varying the following parameters:
- beta, a_last: parameters for the edge connection probabilties. The edge connection probability on level $\ell$ is set to be $\text{a\_last}~\text{beta}^{L-\ell} \log N / N$, with $L$ being the total number of levels.
- num_nodes_per_bottom_community
- example1/2: chose example 1 or example 2
```
$ python3 trial_on_unbalanced.py beta a_last example1/2 num_nodes_per_bottom_community
```

For example to recover one point (at beta = 0.3) in Figure 9 
```
$ python3 trial_on_unbalanced.py 0.3 64 example1 100
```
To recover one point (at beta = 0.3) in Figure 10,
```
$ python3 trial_on_unbalanced.py 0.3 144 example2 100
```

### Deeper BTSBMs (Figure 11)
You can run the following command:
```
$ python3 trial_on_KTSBM.py beta a_last num_nodes_per_bottom_community number_of_levels k-nary
```

For example, to reproduce one point (at beta = 0.3) Figure 11,
```
$ python3 trial_on_KTSBM.py 0.3 81 100 6 2
``` 

## Robustness of linkage to misculstering errors (Figure 3)
```
$ python3 robustness.py
```

## Real datasets
### High shool dataset (Figure 4)
```
$ python3 highschool.py
```
### Military dataset (Figures 5 & 6)
```
$ python3 trial_military_alliance.py
```

### Football dataset (Figure 15)
```
$ python3 football_nt.py
```

## Compare with synthesis (Figure 12)
```
$ python3 compare_bup_synthesis.py
```
The code compares the performance between the bottom-up method and synthesis (Fang, Sijia, and Karl Rohe. "T-Stochastic Graphs." arXiv preprint arXiv:2309.01301 (2023).)

## Aliases
- rbu stands for bottom-up
- rbp stands for top-down