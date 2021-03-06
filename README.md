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
will be somewhat illustrative of what this package is trying to do.

```
python -m graphid.demo.dummy_infr demodata_infr --show
python -m graphid.demo.dummy_infr demodata_infr --num_pccs=25 --show
python -m graphid.demo.dummy_infr demodata_infr --num_pccs=100 --show
```

The first of these commands will generate an image that looks like this: 
![alt text](https://i.imgur.com/CAUJVc5.png "ID-Graph")

The "aid" is an annotation id, the "nid" is a name id. Blue edges mean two annotation match. Red edges mean two annotations don't match. Yellow edges means that two annotations are not comparable (i.e. not enough info to determine if they match or not). Edges that are highlighted are ones flagged by the program for the user to re-check because it inferred that there is an inconsistency in the graph. Edges that are dotted, are flagged by the program as actually between two distinct individuals. 


This README is a mess. 
Why not look at [this Jupyter notebook](notebooks/core_example.ipynb) in the
meantime. (Note as of 2019-June-09 the GitHub viewer seems broken, but 
[this link](https://nbviewer.jupyter.org/github/Erotemic/graphid/blob/master/notebooks/core_example.ipynb)
is working)


# Installation

You should be able to simply `pip install graphid`.
However, you might have an issue with pygraphviz. There also may be issues on Python 3.7 due to use of StopIteration.


# Dependencies

```bash
sudo apt-get install -y graphviz libgraphviz-dev
pip install graphviz
pip install -e . 
```

This project is Python 3.6+, Python 2 is not supported.

If you want to be able to draw the graphs, you must install graphviz, which is
needed by pygraphviz.

I'm currently having trouble getting this to work on windows due to pygraphviz.

Conda can be used to install pygraphviz on windows?
conda install -c marufr pygraphviz
