[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/embodied-computation-group/metadpy/blob/master/LICENSE) [![codecov](https://codecov.io/gh/embodied-computation-group/metadpy/branch/master/graph/badge.svg)](https://codecov.io/gh/embodied-computation-group/metadpy) [![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) [![pip](https://badge.fury.io/py/metadpy.svg)](https://badge.fury.io/py/metadpy)

***

<img src="https://github.com/embodied-computation-group/metadpy/raw/master/docs/source/images/logo.png" align="left" alt="metadpy" height="250" HSPACE=30>

**metadpy** is a Python implementation of standard Bayesian models of behavioural metacognition. It is aimed to provide simple yet powerful functions to compute standard indexes and metrics of signal detection theory (SDT) and metacognitive efficiency (meta-d’ and hierarchical meta-d’). The only input required is a data frame encoding task performances and confidence ratings at the trial level.

**metadpy** is written in Python 3. It uses [Numpy](https://numpy.org/), [Scipy](https://www.scipy.org/) and [Pandas](https://pandas.pydata.org/>) for most of its operation, comprizing meta-d’ estimation using maximum likelihood estimation (MLE). The (Hierarchical) Bayesian modelling is implemented in [Aesara](https://github.com/aesara-devs/aesara) (now renamed [PyTensor](https://github.com/pymc-devs/pytensor) for versions of [pymc](https://docs.pymc.io/>) >=5.0).

* 📖 [Documentation](https://embodied-computation-group.github.io/metadpy/)  
* ✏️ [Tutorials](https://embodied-computation-group.github.io/metadpy/tutorials.html)  

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
* [pymc](https://docs.pymc.io/>) (>=5.0)


# Why metadpy?

metadpy stands for meta-d' (meta-d prime) in Python. meta-d' is a behavioural metric commonly used in consciousness and metacognition research. It is modelled to reflect metacognitive efficiency (i.e the relationship between subjective reports about performances and objective behaviour).

metadpy first aims to be the Python equivalent of the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) (Matlab and R). It tries to make these models available to a broader open-source ecosystem and to ease their use via cloud computing interfaces. One notable difference is that While the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) relies on JAGS for the Bayesian modelling of confidence data (see [**4**]) to analyse task performance and confidence ratings, metadpy is built on the top of [pymc](https://docs.pymc.io/>), and uses Hamiltonina Monte Carlo methods (NUTS).

For an extensive introduction to metadpy, you can navigate the following notebooks that are Python adaptations of the introduction to the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) written in Matlab by Olivia Faul for the [Zurich Computational Psychiatry course](https://github.com/metacoglab/HMeta-d/tree/master/CPC_metacog_tutorial).

# Tutorials

| Notebook | Colab |
| --- | ---|
| What metacognition looks like? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadpy/blob/master/docs/source/examples/1-What%20metacognition%20looks%20like.ipynb)
| Fitting the model (MLE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadpy/blob/master/docs/source/examples/2-Fitting%20the%20model-MLE.ipynb)
| Comparing with the hmetad toolbox | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadpy/blob/master/docs/source/examples/3-Comparison%20with%20the%20hmeta-d%20toolbox.ipynb)

# Examples

| Notebook | Colab |
| --- | ---|
| Subject and group level (MLE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadpy/blob/master/docs/source/examples/Example%201%20-%20Fitting%20MLE%20-%20Subject%20and%20group%20level.ipynb)
| Subject and group level (Bayesian) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadpy/blob/master/docs/source/examples/Example%202%20-%20Fitting%20Bayesian%20-%20Subject%20level%20(pymc).ipynb)

Or just follow the quick tour below.

# Importing data

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


```python
from metadpy.utils import responseSimulation
simulation = responseSimulation(d=1, metad=1.5, c=0, 
                                nRatings=4, nTrials=200)
simulation
```

|    |   Stimuli |   Responses |   Accuracy |   Confidence |   nTrial |   Subject |
|---:|----------:|------------:|-----------:|-------------:|---------:|----------:|
|  0 |         1 |           1 |          1 |            4 |        0 |         0 |
|  1 |         0 |           0 |          1 |            4 |        1 |         0 |
|  2 |         1 |           1 |          1 |            2 |        2 |         0 |
|  3 |         0 |           1 |          0 |            4 |        3 |         0 |
|  4 |         0 |           0 |          1 |            3 |        4 |         0 |

```python
from metadpy.utils import trials2counts
nR_S1, nR_S2 = trials2counts(
    data=simulation, stimuli="Stimuli", accuracy="Accuracy",
    confidence="Confidence", nRatings=4)
```

## Data visualization

You can easily visualize metacognition results using one of the plotting functions. Here, we will use the `plot_confidence` and the `plot_roc` functions to visualize the metacognitive performance of our participant.

```python
import matplotlib.pyplot as plt
from metadpy.plotting import plot_confidence, plot_roc
```

```python
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
plot_confidence(nR_S1, nR_S2, ax=axs[0])
plot_roc(nR_S1, nR_S2, ax=axs[1])
```

![png](./docs/source/images/confidence_ROCAUC.png)

# Signal detection theory (SDT)

```python
from metadpy.sdt import criterion, dprime, rates, roc_auc, scores
```

All metadpy functions are registred as Pandas flavors (see [pandas-flavor](https://pypi.org/project/pandas-flavor/)), which means that the functions can be called as a method from the result data frame.

```python
simulation.criterion()
```

5.551115123125783e-17

```python
simulation.dprime()
```

0.9917006946949065

```python
simulation.rates()
```

(0.69, 0.31)

```python
simulation.roc_auc(nRatings=4)
```

0.695689287238583

```python
simulation.scores()
```

(69, 31, 31, 69)

# Estimating meta dprime using Maximum Likelyhood Estimates (MLE)

```python
from metadpy.mle import metad

metad(
  data=simulation, nRatings=4, stimuli='Stimuli', accuracy='Accuracy',
  confidence='Confidence', verbose=0
  )
```

|    |   dprime |   meta_d |   m_ratio |   m_diff |
|---:|---------:|---------:|----------:|---------:|
|  0 | 0.970635 |  1.45925 |    1.5034 | 0.488613 |

# Estimating meta-dprime using hierarchical Bayesian modeling

## Subject level

```python
import pymc as pm
from metadpy.bayesian import hmetad
```

```python
model, trace = hmetad(
  data=simulation, nRatings=4, stimuli='Stimuli',
  accuracy='Accuracy', confidence='Confidence'
  )
```

Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [c1, d1, meta_d, cS1_hn, cS2_hn]
Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 10 seconds.

```python
import arviz as az
az.plot_trace(trace, var_names=['meta_d', 'cS2', 'cS1']);
```
 
![png](./docs/source/images/trace.png)

```python
az.summary(trace)
```

|        |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:-------|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| meta_d |  1.384 | 0.254 |    0.909 |      1.86 |       0.004 |     0.003 |       3270 |       2980 |       1 |

# References

[1] Maniscalco, B., & Lau, H. (2014). Signal Detection Theory Analysis of Type 1 and Type 2 Data: Meta-d′, Response-Specific Meta-d′, and the Unequal Variance SDT Model. In The Cognitive Neuroscience of Metacognition (pp. 25–66). Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-642-45190-4_3 

[2] Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach for estimating metacognitive sensitivity from confidence ratings. Consciousness and Cognition, 21(1), 422–430. doi:10.1016/j.concog.2011.09.021

[3] Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. Frontiers in Human Neuroscience, 8. https://doi.org/10.3389/fnhum.2014.00443

[4] Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation of metacognitive efficiency from confidence ratings, Neuroscience of Consciousness, 3(1) nix007, https://doi.org/10.1093/nc/nix007
