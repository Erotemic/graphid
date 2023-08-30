The graphid Module
==================


|Pypi| |PypiDownloads| |GithubActions| |Codecov| |ReadTheDocs|


+------------------+----------------------------------------------+
| Read the docs    | https://graphid.readthedocs.io               |
+------------------+----------------------------------------------+
| Github           | https://github.com/Erotemic/graphid          |
+------------------+----------------------------------------------+
| Pypi             | https://pypi.org/project/graphid             |
+------------------+----------------------------------------------+


Graph Identification
====================

A graph algorithm to manage the identification of individuals in a
population using automatic pairwise decision algorithms with a humans in
the loop. It is agnostic to the specific ranking and verification
algorithms. In fact, it can work without a ranking or verification
algorithm, but in that case all reviews will have to be manual, and it
will be difficult to prioritize which pairs of annotations (typically
images) to look at first.

This is the graph identification described in Chapter 5 of
`my thesis <https://github.com/Erotemic/crall-thesis-2017/blob/main/crall-thesis_2017-08-10_compressed.pdf>`_.
Viewing this PDF online can be slow, so I’ve linked there raw text
`here <https://github.com/Erotemic/crall-thesis-2017/blob/main/chapter5-graphid.tex>`_.

General Information
===================

While this repo is functional, I don't maintain it as often as I do for other
libraries that I regularly use. Its use case is fairly niche.

Helpful commands I’m currently using in development and debugging.
Perhaps they will be somewhat illustrative of what this package is
trying to do.

::

   python -m graphid.demo.dummy_infr demodata_infr --show
   python -m graphid.demo.dummy_infr demodata_infr --num_pccs=25 --show
   python -m graphid.demo.dummy_infr demodata_infr --num_pccs=100 --show

The first of these commands will generate an image that looks like this:


.. image:: https://i.imgur.com/CAUJVc5.png
   :height: 300px
   :align: left

The "aid" is an annotation id, the "nid" is a name id. Blue edges mean
two annotation match. Red edges mean two annotations don’t match. Yellow
edges means that two annotations are not comparable (i.e. not enough
info to determine if they match or not). Edges that are highlighted are
ones flagged by the program for the user to re-check because it inferred
that there is an inconsistency in the graph. Edges that are dotted, are
flagged by the program as actually between two distinct individuals.

This README is a mess. Why not look at `this Jupyter
notebook <notebooks/core_example.ipynb>`__ in the meantime. (Note as of
2019-June-09 the GitHub viewer seems broken, but `this
link <https://nbviewer.jupyter.org/github/Erotemic/graphid/blob/main/notebooks/core_example.ipynb>`_
is working)

Installation
============

You should be able to simply ``pip install graphid``. However, you might
have an issue with pygraphviz. There also may be issues on Python 3.7
due to use of StopIteration.

Dependencies
============

.. code:: bash

   sudo apt-get install -y graphviz libgraphviz-dev
   pip install pygraphviz

See `pygraphviz install details <https://github.com/pygraphviz/pygraphviz/blob/main/INSTALL.txt>`_
for more information and instructions for non-debian systems.

This project is Python 3.6+, Python 2 is not supported.

If you want to be able to draw the graphs, you must install graphviz,
which is needed by pygraphviz.

I’m currently having trouble getting this to work on windows due to
pygraphviz.

Conda can be used to install pygraphviz on Windows?
``conda install -c marufr pygraphviz``. Not sure if this works. I recommend
avoiding conda if possible, but Windows is one place where I think it has a
good use-case.



.. |Pypi| image:: https://img.shields.io/pypi/v/graphid.svg
    :target: https://pypi.python.org/pypi/graphid

.. |PypiDownloads| image:: https://img.shields.io/pypi/dm/graphid.svg
    :target: https://pypistats.org/packages/graphid

.. |GithubActions| image:: https://github.com/Erotemic/graphid/actions/workflows/tests.yml/badge.svg?branch=main
    :target: https://github.com/Erotemic/graphid/actions?query=branch%3Amain

.. |Codecov| image:: https://codecov.io/github/Erotemic/graphid/badge.svg?branch=main&service=github
    :target: https://codecov.io/github/Erotemic/graphid?branch=main

.. |ReadTheDocs| image:: https://readthedocs.org/projects/graphid/badge/?version=latest
    :target: http://graphid.readthedocs.io/en/latest/
