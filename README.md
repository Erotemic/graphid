[![Travis](https://img.shields.io/travis/Erotemic/graphid/master.svg?label=Travis%20CI)](https://travis-ci.org/Erotemic/graphid)
[![Codecov](https://codecov.io/github/Erotemic/graphid/badge.svg?branch=master&service=github)](https://codecov.io/github/Erotemic/graphid?branch=master)
[![Appveyor](https://ci.appveyor.com/api/projects/status/github/Erotemic/graphid?svg=True)](https://ci.appveyor.com/project/Erotemic/graphid/branch/master)
[![Pypi](https://img.shields.io/pypi/v/graphid.svg)](https://pypi.python.org/pypi/graphid)

# Graph Identification

This is the graph identification described in Chapter 5 of [my thesis](https://github.com/Erotemic/crall-thesis-2017/blob/master/crall-thesis_2017-08-10_compressed.pdf). Viewing this PDF online can be slow, so I've linked there raw text [here](https://github.com/Erotemic/crall-thesis-2017/blob/master/chapter5-graphid.tex).


# General Information

Helpful commands I'm currently using in development and debugging. Perhaps they
will be someone illustrative of what this package is trying to do.

```
python -m graphid.demo.dummy_infr demodata_infr --show
python -m graphid.demo.dummy_infr demodata_infr --num_pccs=25 --show
python -m graphid.demo.dummy_infr demodata_infr --num_pccs=100 --show
```

This README is a mess. Why not look at [this Jupyter
notebook](notebooks/core_example.ipynb) in the meantime.



# Dependencies

This project is Python 3.6+, Python 2 is not supported.

I'm currently having trouble getting this to work on windows due to pygraphviz.

Conda can be used to install pygraphviz on windows?
conda install -c marufr pygraphviz