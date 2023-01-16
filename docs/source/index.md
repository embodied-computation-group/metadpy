---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "_I8iFrneRXFP"}

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/LegrandNico/metadpy/blob/master/LICENSE) [![pip](https://badge.fury.io/py/metadpy.svg)](https://badge.fury.io/py/metadpy) [![travis](https://travis-ci.com/LegrandNico/metadpy.svg?branch=master)](https://travis-ci.com/LegandNico/metadpy) [![codecov](https://codecov.io/gh/LegrandNico/metadpy/branch/master/graph/badge.svg)](https://codecov.io/gh/LegrandNico/metadpy) [![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

***

<img src="https://github.com/LegrandNico/metadpy/raw/master/docs/source/images/logo.png" align="left" alt="metadpy" height="250" HSPACE=30>

**metadpy** is a Python implementation of standard Bayesian models of behavioural metacognition. It is aimed to provide simple yet powerful functions to compute standard indexes and metrics of signal detection theory (SDT) and metacognitive efficiency (meta-d’ and hierarchical meta-d’) {cite:p}`fleming:2014,fleming:2017,maniscalo:2014,maniscalo:2012`. The only input required is a data frame encoding task performances and confidence ratings at the trial level.

**metadpy** is written in Python 3. It uses [Numpy](https://numpy.org/), [Scipy](https://www.scipy.org/) and [Pandas](https://pandas.pydata.org/>) for most of its operation, comprizing meta-d’ estimation using maximum likelihood estimation (MLE). The (Hierarchical) Bayesian modelling is implemented in [Aesara](https://github.com/aesara-devs/aesara) (now renamed [PyTensor](https://github.com/pymc-devs/pytensor) for versions of [pymc](https://docs.pymc.io/>) >5.0).

# Installation

The package can be installed using pip:

```shell
pip install metadpy
```

For most of the operations, the following packages are required:

* [Numpy](https://numpy.org/) (>=1.15)
* [Scipy](https://www.scipy.org/) (>=1.3.0)
* [Pandas](https://pandas.pydata.org/>) (>=0.24)
* [Matplotlib](https://matplotlib.org/) (>=3.0.2)
* [Seaborn](https://seaborn.pydata.org/) (>=0.9.0)

Bayesian models will require:

* [PyTensor](https://github.com/pymc-devs/pytensor)
* [pymc](https://docs.pymc.io/>) >5.0)

+++ {"id": "Ptr2p3eWTxMX"}

# Why metadpy?

metadpy stands for meta-d' (meta-d prime) in Python. meta-d' is a behavioural metric commonly used in consciousness and metacognition research. It is modelled to reflect metacognitive efficiency (i.e the relationship between subjective reports about performances and objective behaviour).

metadpy first aims to be the Python equivalent of the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) (Matlab and R). It tries to make these models available to a broader open-source ecosystem and to ease their use via cloud computing interfaces. One notable difference is that While the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) relies on JAGS for the Bayesian modelling of confidence data (see [**4**]) to analyse task performance and confidence ratings, metadpy is built on the top of [pymc](https://docs.pymc.io/>), and uses Hamiltonina Monte Carlo methods (NUTS).

For an extensive introduction to metadpy, you can navigate the following notebooks that are Python adaptations of the introduction to the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) written in Matlab by Olivia Faul for the [Zurich Computational Psychiatry course](https://github.com/metacoglab/HMeta-d/tree/master/CPC_metacog_tutorial).

✏️ [Tutorials and examples](https://embodied-computation-group.github.io/metadpy/tutorials.html)  

+++ {"id": "w0EklNnNf6Ms"}

# Importing data

+++ {"id": "69r9Nrw6dsP0"}

Classical metacognition experiments contain two phases: task performance and confidence ratings. The task performance could for example be the ability to distinguish the presence of a dot on the screen. By relating trials where stimuli are present or absent and the response provided by the participant (Can you see the dot: yes/no), it is possible to obtain the accuracy. The confidence rating is proposed to the participant when the response is made and should reflect how certain the participant is about his/her judgement.

An ideal observer would always associate very high confidence ratings with correct task-I responses, and very low confidence ratings with an incorrect task-1 response, while a participant with a low metacognitive efficiency will have a more mixed response pattern.

A minimal metacognition dataset will therefore consist in a data frame populated with 5 columns:
* `Stimuli`: Which of the two stimuli was presented [0 or 1].
* `Response`: The response made by the participant [0 or 1].
* `Accuracy`: Was the participant correct? [0 or 1].
* `Confidence`: The confidence level [can be continuous or discrete rating].
* `ntrial`: The trial number.

Due to the logical dependence between the `Stimuli`, `Responses` and `Accuracy` columns, in practice only two of those columns are necessary, the third being deduced from the others. Most of the functions in `metadpy` will accept DataFrames containing only two of these columns, and will automatically infer the missing information. Similarly, as the metacognition models described here does not incorporate the temporal dimension, the trial number is optional. 

`metadpy` includes a simulation function that will let you create one such data frame for one or many participants and condition, controlling for a variety of parameters. Here, we will simulate 200 trials from  participant having `d=1` and `c=0` (task performances) and a `meta-d=1.5` (metacognitive sensibility). The confidence ratings were provided using a 1-to-4 rating scale.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 424
id: wOPOtKuIdrzD
outputId: 511facb6-f4a8-4ca3-cb91-e877e49aa643
---
from metadpy.utils import responseSimulation

simulation = responseSimulation(d=1, metad=2.0, c=0, nRatings=4, nTrials=5000)
simulation.head()
```

```{code-cell} ipython3
:id: F2An4oGxgbz2

from metadpy.utils import trials2counts

nR_S1, nR_S2 = trials2counts(
    data=simulation,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
    nRatings=4,
)
```

```{code-cell} ipython3
nR_S1, nR_S2
```

+++ {"id": "X_5VeOp-f8Hz"}

## Data visualization

+++ {"id": "DScjwm4QgERk"}

You can easily visualize metacognition results using one of the plotting functions. Here, we will use the `plot_confidence` and the `plot_roc` functions to visualize the metacognitive performance of our participant.

```{code-cell} ipython3
:id: 5ulkBLZWf-zz

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from metadpy.plotting import plot_confidence, plot_roc

sns.set_context("talk")
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 396
id: MxU7AvBbgMwc
outputId: 954a428f-be2e-40c4-9580-56e65d870fbd
---
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
plot_confidence(nR_S1, nR_S2, ax=axs[0])
plot_roc(nR_S1, nR_S2, ax=axs[1])
sns.despine()
```

+++ {"id": "epcdJnLUdT8e"}

# Signal detection theory (SDT)

+++ {"id": "EqC4DJN_6KHG"}

All metadpy functions are registred as Pandas flavors (see [pandas-flavor](https://pypi.org/project/pandas-flavor/)), which means that the functions can be called as a method from the result data frame. When using the default columns names (`Stimuli`, `Response`, `Accuracy`, `Confidence`), this significantly reduces the length of the function call, making your code more clean and readable.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: peZDc-xv5Qm2
outputId: c344e8f7-9c7b-4e35-c965-0dcf8f3303c6
---
simulation.criterion()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: l8IppukU5QzU
outputId: 00f03f29-3c2a-4857-bc92-a65a2ce9c396
---
simulation.dprime()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 6Ki5MBP-5aO5
outputId: 368213aa-4dc8-44cd-b268-0e4b8ade3fb8
---
simulation.rates()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: XU1T7YLW5jHT
outputId: 37688d92-b4a0-447c-ad77-22059281999a
---
simulation.roc_auc(nRatings=4)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: Cr0-o9DN5Q9h
outputId: d1df3cb6-4328-4a97-ebc5-5064d6c1b9ef
---
simulation.scores()
```

+++ {"id": "ntThWcGodWNU"}

# Estimating meta dprime using Maximum Likelyhood Estimates (MLE)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: TrsF_KUYdZzf
outputId: 72830038-8aa7-4a13-ba29-32aa22576ca0
---
from metadpy.mle import metad

results = metad(
    data=simulation,
    nRatings=4,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
    verbose=0,
)
results
```

+++ {"id": "d6n1XEgAdbId"}

# Estimating meta-d' using Bayesian modeling

+++ {"id": "16zp_Md4qM-X"}

## Subject level

```{code-cell} ipython3
:id: I08VOEauqbsc

from metadpy.bayesian import hmetad
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 222
id: eUCVaD0udcBc
outputId: 58d8d546-d04e-428d-86e0-b9a08d3715a3
---
model, trace = hmetad(
    data=simulation,
    nRatings=4,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 457
id: 4GWUwuihsfpf
outputId: f10724a7-4af2-4c9e-b082-0dc7b9911c19
---
az.plot_trace(trace, var_names=["meta_d", "cS2", "cS1"]);
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 269
id: LXP87UvjsR6O
outputId: d817b6e0-951e-454b-f614-4fc7890d3736
---
az.summary(trace, var_names=["meta_d", "cS2", "cS1"])
```

<img src = "https://raw.githubusercontent.com/embodied-computation-group/metadpy/master/docs/source/images/LabLogo.png" height ="100"><img src = "https://raw.githubusercontent.com/embodied-computation-group/metadpy/master/docs/source/images/AU.png" height ="100">

```{toctree}
---
hidden:
---
API <api.rst>
Tutorials <tutorials.md>
Cite <cite.md>
References <references.md>
```

```{code-cell} ipython3

```
