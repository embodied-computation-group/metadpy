---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="_I8iFrneRXFP" -->
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/LegrandNico/metadPy/blob/master/LICENSE) [![pip](https://badge.fury.io/py/metadPy.svg)](https://badge.fury.io/py/metadPy) [![travis](https://travis-ci.com/LegrandNico/metadPy.svg?branch=master)](https://travis-ci.com/LegandNico/metadPy) [![codecov](https://codecov.io/gh/LegrandNico/metadPy/branch/master/graph/badge.svg)](https://codecov.io/gh/LegrandNico/metadPy) [![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

***

<img src="https://github.com/LegrandNico/metadPy/raw/master/images/logo.png" align="left" alt="metadPy" height="250" HSPACE=30>

**metadPy** is an open-source Python package for cognitive modelling of behavioural data with a focus on metacognition. It is aimed to provide simple yet powerful functions to compute standard index and metric of signal detection theory (SDT) and metacognitive efficiency (meta-d’ using both MLE and Bayesian estimations) [**1**, **2**, **3**]. The only input required is a data frame encoding task performances and confidence ratings at the trial level.

**metadPy** is written in Python 3. It uses [Numpy](https://numpy.org/), [Scipy](https://www.scipy.org/) and [Pandas](https://pandas.pydata.org/>) for most of its operation, comprizing meta-d’ estimation using maximum likelihood estimation (MLE). The (Hierarchical) Bayesian modelling of meta-d’ and m-ratio [**4**] is based either on [PyMC v4](https://docs.pymc.io/>).

# Installation

metadPy can be installed using pip:

```shell
pip install git+https://github.com/LegrandNico/metadPy.git
```

For most of the operations, the following packages are required:

* [Numpy](https://numpy.org/) (>=1.15)
* [Scipy](https://www.scipy.org/) (>=1.3.0)
* [Pandas](https://pandas.pydata.org/>) (>=0.24)
* [Matplotlib](https://matplotlib.org/) (>=3.0.2)
* [Seaborn](https://seaborn.pydata.org/) (>=0.9.0)

For Bayesian modelling you will need:

* [PyMC](https://docs.pymc.io/>) (>=4.0.0).
<!-- #endregion -->

<!-- #region id="Ptr2p3eWTxMX" -->
# Why metadPy?

metadPy stands for meta-d' (meta-d prime) in Python. meta-d' is a behavioural metric commonly used in consciousness and metacognition research. It is modelled to reflect metacognitive efficiency (i.e the relationship between subjective reports about performances and objective behaviour).

metadPy first aims to be the Python equivalent of the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) (Matlab and R). It tries to make these models available to a broader open-source ecosystem and to ease their use via cloud computing interfaces. One notable difference is that While the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) relies on JAGS for the Bayesian modelling of confidence data (see [**4**]) to analyse task performance and confidence ratings, metadPy is based on [JAX](https://jax.readthedocs.io/en/latest/) and [Numpyro](https://num.pyro.ai/en/latest/index.html#), which can easily be parallelized, flexibly uses CPU, GPU or TPU and offers a broader variety of MCMC sampling algorithms (comprising NUTS).

For an extensive introduction to metadPy, you can navigate the following notebooks that are Python adaptations of the introduction to the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) written in Matlab by Olivia Faul for the [Zurich Computational Psychiatry course](https://github.com/metacoglab/HMeta-d/tree/master/CPC_metacog_tutorial).

## Examples 

| Notebook | Colab | nbViewer |
| --- | ---| --- |
| Example 1 - Fitting MLE - Subject and group level | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadPy/blob/master/docs/notebooks/Example%201%20-%20Fitting%20MLE%20-%20Subject%20and%20group%20level.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/embodied-computation-group/metadPy/blob/master/docs/notebooks/Example%201%20-%20Fitting%20MLE%20-%20Subject%20and%20group%20level.ipynb)
| Example 2 - Fitting Bayesian - Subject level (pymc) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadPy/blob/master/docs/notebooks/Example%202%20-%20Fitting%20Bayesian%20-%20Subject%20level%20(pymc).ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/embodied-computation-group/metadPy/blob/master/docs/notebooks/Example%202%20-%20Fitting%20Bayesian%20-%20Subject%20level%20(pymc).ipynb)


## Tutorials

| Notebook | Colab | nbViewer |
| --- | ---| --- |
| 1. What metacognition looks like? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadPy/blob/master/docs/notebooks/1%20-%20What%20metacognition%20looks%20like.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/embodied-computation-group/metadPy/blob/master/docs/notebooks/1%20-%20What%20metacognition%20looks%20like.ipynb)
| 2. Fitting the model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadPy/blob/master/docs/notebooks/2%20-%20Fitting%20the%20model.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/embodied-computation-group/metadPy/blob/master/docs/notebooks/2%20-%20Fitting%20the%20model.ipynb)
| 3. Comparison with the HMeta-d toolbox | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadPy/blob/master/docs/notebooks/3-Comparison%20with%20the%20hmeta-d%20toolbox.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/embodied-computation-group/metadPy/blob/master/docs/notebooks/3-Comparison%20with%20the%20hmeta-d%20toolbox.ipynb)
<!-- #endregion -->

<!-- #region id="w0EklNnNf6Ms" -->
# Importing data
<!-- #endregion -->

<!-- #region id="69r9Nrw6dsP0" -->
Classical metacognition experiments contain two phases: task performance and confidence ratings. The task performance could for example be the ability to distinguish the presence of a dot on the screen. By relating trials where stimuli are present or absent and the response provided by the participant (Can you see the dot: yes/no), it is possible to obtain the accuracy. The confidence rating is proposed to the participant when the response is made and should reflect how certain the participant is about his/her judgement.

An ideal observer would always associate very high confidence ratings with correct task-I responses, and very low confidence ratings with an incorrect task-1 response, while a participant with a low metacognitive efficiency will have a more mixed response pattern.

A minimal metacognition dataset will therefore consist in a data frame populated with 5 columns:
* `Stimuli`: Which of the two stimuli was presented [0 or 1].
* `Response`: The response made by the participant [0 or 1].
* `Accuracy`: Was the participant correct? [0 or 1].
* `Confidence`: The confidence level [can be continuous or discrete rating].
* `ntrial`: The trial number.

Due to the logical dependence between the `Stimuli`, `Responses` and `Accuracy` columns, in practice only two of those columns are necessary, the third being deduced from the others. Most of the functions in `metadPy` will accept DataFrames containing only two of these columns, and will automatically infer the missing information. Similarly, as the metacognition models described here does not incorporate the temporal dimension, the trial number is optional. 

`metadPy` includes a simulation function that will let you create one such data frame for one or many participants and condition, controlling for a variety of parameters. Here, we will simulate 200 trials from  participant having `d=1` and `c=0` (task performances) and a `meta-d=1.5` (metacognitive sensibility). The confidence ratings were provided using a 1-to-4 rating scale.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 424} id="wOPOtKuIdrzD" outputId="511facb6-f4a8-4ca3-cb91-e877e49aa643"
from metadPy.utils import responseSimulation

simulation = responseSimulation(d=1, metad=2.0, c=0, nRatings=4, nTrials=5000)
simulation.head()
```

```python id="F2An4oGxgbz2"
from metadPy.utils import trials2counts

nR_S1, nR_S2 = trials2counts(
    data=simulation,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
    nRatings=4,
)
```

```python
nR_S1, nR_S2
```

<!-- #region id="X_5VeOp-f8Hz" -->
## Data visualization
<!-- #endregion -->

<!-- #region id="DScjwm4QgERk" -->
You can easily visualize metacognition results using one of the plotting functions. Here, we will use the `plot_confidence` and the `plot_roc` functions to visualize the metacognitive performance of our participant.
<!-- #endregion -->

```python id="5ulkBLZWf-zz"
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from metadPy.plotting import plot_confidence, plot_roc

sns.set_context("talk")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 396} id="MxU7AvBbgMwc" outputId="954a428f-be2e-40c4-9580-56e65d870fbd"
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
plot_confidence(nR_S1, nR_S2, ax=axs[0])
plot_roc(nR_S1, nR_S2, ax=axs[1])
sns.despine()
```

<!-- #region id="epcdJnLUdT8e" -->
# Signal detection theory (SDT)
<!-- #endregion -->

<!-- #region id="EqC4DJN_6KHG" -->
All metadPy functions are registred as Pandas flavors (see [pandas-flavor](https://pypi.org/project/pandas-flavor/)), which means that the functions can be called as a method from the result data frame. When using the default columns names (`Stimuli`, `Response`, `Accuracy`, `Confidence`), this significantly reduces the length of the function call, making your code more clean and readable.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="peZDc-xv5Qm2" outputId="c344e8f7-9c7b-4e35-c965-0dcf8f3303c6"
simulation.criterion()
```

```python colab={"base_uri": "https://localhost:8080/"} id="l8IppukU5QzU" outputId="00f03f29-3c2a-4857-bc92-a65a2ce9c396"
simulation.dprime()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6Ki5MBP-5aO5" outputId="368213aa-4dc8-44cd-b268-0e4b8ade3fb8"
simulation.rates()
```

```python colab={"base_uri": "https://localhost:8080/"} id="XU1T7YLW5jHT" outputId="37688d92-b4a0-447c-ad77-22059281999a"
simulation.roc_auc(nRatings=4)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Cr0-o9DN5Q9h" outputId="d1df3cb6-4328-4a97-ebc5-5064d6c1b9ef"
simulation.scores()
```

<!-- #region id="ntThWcGodWNU" -->
# Estimating meta dprime using Maximum Likelyhood Estimates (MLE)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="TrsF_KUYdZzf" outputId="72830038-8aa7-4a13-ba29-32aa22576ca0"
from metadPy.mle import metad

metad = metad(
    data=simulation,
    nRatings=4,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
    verbose=0,
)
print(f'meta-d\' = {str(metad["meta_d"])}')
```

<!-- #region id="d6n1XEgAdbId" -->
# Estimating meta-dprime using hierarchical Bayesian modeling
<!-- #endregion -->

<!-- #region id="16zp_Md4qM-X" -->
## Subject level
<!-- #endregion -->

```python id="I08VOEauqbsc"
from metadPy.bayesian import hmetad
```

```python colab={"base_uri": "https://localhost:8080/", "height": 222} id="eUCVaD0udcBc" outputId="58d8d546-d04e-428d-86e0-b9a08d3715a3"
model, trace = hmetad(
    data=simulation,
    nRatings=4,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 457} id="4GWUwuihsfpf" outputId="f10724a7-4af2-4c9e-b082-0dc7b9911c19"
az.plot_trace(trace, var_names=["meta_d", "cS2", "cS1"]);
```

```python colab={"base_uri": "https://localhost:8080/", "height": 269} id="LXP87UvjsR6O" outputId="d817b6e0-951e-454b-f614-4fc7890d3736"
az.summary(trace, var_names=["meta_d", "cS2", "cS1"])
```

<!-- #region id="vFHmSwwzRXFo" -->
# References
<!-- #endregion -->

<!-- #region id="AndtVDjLRXFo" -->
[1] Maniscalco, B., & Lau, H. (2014). Signal Detection Theory Analysis of Type 1 and Type 2 Data: Meta-d′, Response-Specific Meta-d′, and the Unequal Variance SDT Model. In The Cognitive Neuroscience of Metacognition (pp. 25–66). Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-642-45190-4_3 

[2] Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. Frontiers in Human Neuroscience, 8. https://doi.org/10.3389/fnhum.2014.00443

[3] Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation of metacognitive efficiency from confidence ratings, Neuroscience of Consciousness, 3(1) nix007, https://doi.org/10.1093/nc/nix007
<!-- #endregion -->

# Watermark

```python
%load_ext watermark
%watermark -n -u -v -iv -w -p metadPy,pymc
```

```python

```
