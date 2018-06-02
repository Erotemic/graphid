[![Travis](https://img.shields.io/travis/Erotemic/graphid/master.svg?label=Travis%20CI)](https://travis-ci.org/Erotemic/graphid)
[![Codecov](https://codecov.io/github/Erotemic/graphid/badge.svg?branch=master&service=github)](https://codecov.io/github/Erotemic/graphid?branch=master)
[![Appveyor](https://ci.appveyor.com/api/projects/status/github/Erotemic/graphid?svg=True)](https://ci.appveyor.com/project/Erotemic/graphid/branch/master)
[![Pypi](https://img.shields.io/pypi/v/graphid.svg)](https://pypi.python.org/pypi/graphid)

# Graph Identification

A graph algorithm to manage the identification of individuals in a population
using automatic pairwise decision algorithms with a humans in the loop.  It is
agnostic to the specific ranking and verification algorithms. In fact, it can
work without a ranking or verification algorithm, but in that case all reviews
will have to be manual, and it will be difficult to prioritize which pairs of
annotations (typically images) to look at first.

This is the graph identification described in Chapter 5 of [my thesis](https://github.com/Erotemic/crall-thesis-2017/blob/master/crall-thesis_2017-08-10_compressed.pdf). Viewing this PDF online can be slow, so I've linked there raw text [here](https://github.com/Erotemic/crall-thesis-2017/blob/master/chapter5-graphid.tex).


# General Information

This repo is currently a work in progress. 

Helpful commands I'm currently using in development and debugging. Perhaps they
will be someone illustrative of what this package is trying to do.

```
python -m graphid.demo.dummy_infr demodata_infr --show
python -m graphid.demo.dummy_infr demodata_infr --num_pccs=25 --show
python -m graphid.demo.dummy_infr demodata_infr --num_pccs=100 --show
```

This README is a mess. Why not look at [this Jupyter
notebook](notebooks/core_example.ipynb) in the meantime.


# Installation

Once this package becomes stable you can install via `pip install graphid`.
However, this will currently give you an older version of the project I
uploaded to reserve the name.


# Dependencies

This project is Python 3.6+, Python 2 is not supported.

If you want to be able to draw the graphs, you must install graphviz, which is
needed by pygraphviz.

I'm currently having trouble getting this to work on windows due to pygraphviz.

Conda can be used to install pygraphviz on windows?
conda install -c marufr pygraphviz



# Technical Information

The `AnnotInference` object consists of several `networkx` data structures. 
By carefully updating the information in these backend "bookkeeping"
structures we are able to quickly access interesting information about the
status of the identification process. For example we gain fast access to these
attributes:

* The number of uniquely identified individuals. Assuming no mistakes were
  made, this is a lower bound on true the number of individuals in the data.

* If a labeling inconsistency has occurred, and which individual it belongs to.

* The next most likely **positive** edge, that cannot be automatically decided
  on.


## AnnotInference core attributes


### Networkx Graph Data Structures


The core `networkx` data structures view annotations as nodes and pairwise
decisions between annotations as edges: 

* `infr.graph` - The core graph structure which contains all edges with labeled
  types
* `infr.review_graphs` - which maps edge labels to graphs that only contain
  edges with those labels. Also note that in the code, the graphs in this dict
  are often referenced via convenience accessors. These keys and their aliases
  are:
    * `POSTV` -  `infr.positive_graph` - the graph containing only positive edges. This is the most important graph used for identifying the PCCs.
    * `NEGTV` -  `infr.neg_graph` - the graph containing only negative edges. 
    * `INCMP` -  `infr.incomp_graph` - the graph containing only incomparable edges
    * `UNKWN` -  `infr.unknown_graph` - the graph containing edges where a user was unable to make a positive, negative, or incomparable decision.
    * `UNREV` -  `infr.unreviewed_graph` - the graph containing edges that have been suggested by the ranking algorithm, but have not yet been reviewed. Most, but not all, of these edges will be in the priority queue. 

* `infr.recover_graph` - This graph only contains the annotation nodes and
  positive edges for **inconsistent** PCCs (i.e. that have at least one
internal negative edge).

Additionally, we maintain "meta" `networkx` data structure where the nodes
are individuals (existing PCCs) and the edges represent weighted negative
decisions. In the code these are referenced as:

* `infr.neg_metagraph` - Each node is a PCC and each edge has a weight where
  the weight is the number of negative decisions made between those PCCs.

* `infr.neg_redun_metagraph` - (POSSIBLY DEPRECATE?) This is the same as
`infr.neg_metagraph`, but only contains an edge if the number of negative edges
is greater than a threshold. 


### Other data structures

`infr.queue` - The priority queue which holds the next edges to be reviewed

`infr.refresh` - This custom object lets us estimate the probability that any
undiscovered positive reviews remain. 



## Theoretical Properties

### Guarantees for incomplete graphs

A graph is incomplete if it is not positive and negative redundant. Another way
to think about it is that a graph is incomplete if the meta-graph --- where
names are nodes edges are indicate a weighted negative relationship --- is not
a **complete graph**. Similarly, a inference graph is complete iff (1): its
PCCs are all positive redundant and (2) the meta-graph is complete. Obviously
the goal of `graphid` is to complete the graph, but actually getting to that
state is usually not feasible. Even so, we can still make the following
guarantees about incomplete graphs:


Assuming no mistakes were made, can obtain the following information, even in
an incomplete state:

    * An upper and lower bound on the number of unique individuals.
        - Lower: the number of PCCs

If mistakes were made, we can still say:
    * If there is a PCC with an internal inconsistency, and the path where the
      inconsistency occurs.


K-positive redundancy is essentially asking the question: 
What guarantees can I get if: 
(a) I assume there are at least K incorrect reviews within a PCC.
(b) I assume there are at least K correct reviews within a PCC.
(c) I assume there are at most K incorrect reviews within a PCC.
(d) I assume there are at most K correct reviews within a PCC.


If K=2, then I think: "in at least one of these cases I think we can guarantee
that the PCC is correct, because if it wasn't it would trigger the
inconsistency criterion."


