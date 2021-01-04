.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
  :target: https://github.com/LegrandNico/metadPy/blob/master/LICENSE

.. image:: https://badge.fury.io/py/metadPy.svg
  :target: https://badge.fury.io/py/metadPy

.. image:: https://travis-ci.com/LegrandNico/metadPy.svg?branch=master
  :target: https://travis-ci.com/LegandNico/metadPy

.. image:: https://codecov.io/gh/LegrandNico/metadPy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/LegrandNico/metadPy

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black

================

.. image::  https://github.com/LegrandNico/metadPy/blob/master/images/logo.png

================

metadPy
=======

Measuring metacognition with Python.

metadPy implement standard Signal Detection Theory metrics as well as MLE [#]_ and hierarchical bayesian estimates of meta-d' [#]_.

It is build on the top of `Pandas <https://pandas.pydata.org/>`_ and uses `PyMC3 <https://docs.pymc.io/>`_ to let you fit a model with a single line of code.

Installation
============

metadPy can be installed using pip:

.. code-block:: shell

  pip install git+https://github.com/LegrandNico/metadPy.git

The following packages are required:

* Numpy (>=1.15)
* SciPy (>=1.3.0)
* Pandas (>=0.24)
* Matplotlib (>=3.0.2)
* Seaborn (>=0.9.0)
* PyMC3 (>=3.8)

Quick start
===========

```python
from metadPy import load_dataset
dataset = load_dataset('rm)
dataset.head()
```

Signal Detection Theory (SDT)
-----------------------------


Estimating meta-dprime using Maximum Likelyhood Estimates (MLE)
---------------------------------------------------------------


Estimating meta-dprime using hierarchical Bayesian modeling
-----------------------------------------------------------

Single subject model
""""""""""""""""""""

Repeated measures model (two-way)
"""""""""""""""""""""""""""""""""


References
==========

.. [#] Maniscalco, B., & Lau, H. (2014). Signal Detection Theory Analysis of Type 1 and Type 2 Data: Meta-d′, Response-Specific Meta-d′, and the Unequal Variance SDT Model. In The Cognitive Neuroscience of Metacognition (pp. 25–66). Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-642-45190-4_3 

.. [#] Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. Frontiers in Human Neuroscience, 8. https://doi.org/10.3389/fnhum.2014.00443

.. [#] Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation of metacognitive efficiency from confidence ratings, Neuroscience of Consciousness, 3(1) nix007, https://doi.org/10.1093/nc/nix007
