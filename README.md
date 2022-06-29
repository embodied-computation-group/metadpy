[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/LegrandNico/metadPy/blob/master/LICENSE) [![pip](https://badge.fury.io/py/metadPy.svg)](https://badge.fury.io/py/metadPy) [![travis](https://travis-ci.com/LegrandNico/metadPy.svg?branch=master)](https://travis-ci.com/LegandNico/metadPy) [![codecov](https://codecov.io/gh/LegrandNico/metadPy/branch/master/graph/badge.svg)](https://codecov.io/gh/LegrandNico/metadPy) [![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

***

<img src="https://github.com/LegrandNico/metadPy/raw/master/images/logo.png" align="left" alt="metadPy" height="250" HSPACE=30>

**metadPy** is an open-source Python package for cognitive modelling of behavioural data with a focus on metacognition. It is aimed to provide simple yet powerful functions to compute standard index and metric of signal detection theory (SDT) and metacognitive efficiency (meta-d’ using both MLE and Bayesian estimations) [**1**, **2**, **3**]. The only input required is a data frame encoding task performances and confidence ratings at the trial level.

**metadPy** is written in Python 3. It uses [Numpy](https://numpy.org/), [Scipy](https://www.scipy.org/) and [Pandas](https://pandas.pydata.org/>) for most of its operation, comprizing meta-d’ estimation using maximum likelihood estimation (MLE). The (Hierarchical) Bayesian modelling of meta-d’ and m-ratio [**4**] is based either on [JAX](https://jax.readthedocs.io/en/latest/) and [Numpyro](https://num.pyro.ai/en/latest/index.html#), or on [pymc](https://docs.pymc.io/>).

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

For Bayesian modelling you will either need:

* [Numpyro](https://num.pyro.ai/en/latest/index.html#introductory-tutorials) (>=0.8.0) - also requiers [JAX](https://jax.readthedocs.io/en/latest/)

  *or*

* [PyMC](https://docs.pymc.io/>) (>=4.0.0).

# Why metadPy?

metadPy stands for meta-d' (meta-d prime) in Python. meta-d' is a behavioural metric commonly used in consciousness and metacognition research. It is modelled to reflect metacognitive efficiency (i.e the relationship between subjective reports about performances and objective behaviour).

metadPy first aims to be the Python equivalent of the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) (Matlab and R). It tries to make these models available to a broader open-source ecosystem and to ease their use via cloud computing interfaces. One notable difference is that While the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) relies on JAGS for the Bayesian modelling of confidence data (see [**4**]) to analyse task performance and confidence ratings, metadPy is based on [JAX](https://jax.readthedocs.io/en/latest/) and [Numpyro](https://num.pyro.ai/en/latest/index.html#), which can easily be parallelized, flexibly uses CPU, GPU or TPU and offers a broader variety of MCMC sampling algorithms (comprising NUTS).

For an extensive introduction to metadPy, you can navigate the following notebooks that are Python adaptations of the introduction to the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) written in Matlab by Olivia Faul for the [Zurich Computational Psychiatry course](https://github.com/metacoglab/HMeta-d/tree/master/CPC_metacog_tutorial).

## Examples 

| Notebook | Colab | nbViewer |
| --- | ---| --- |
| 1. Estimating meta-d' using MLE (subject and group level) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadPy/blob/master/notebooks/Example%201%20-%20Fitting%20MLE%20-%20Subject%20and%20group%20level.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/embodied-computation-group/metadPy/blob/master/notebooks/Example%201%20-%20Fitting%20MLE%20-%20Subject%20and%20group%20level.ipynb)
| 2. Estimating meta-d' (single subject) using Bayesian modelling - Numpyro | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadPy/blob/master/notebooks/Example%202%20-%20Fitting%20Bayesian%20-%20Subject%20level%20(numpyro).ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/embodied-computation-group/metadPy/blob/master/notebooks/Example%202%20-%20Fitting%20Bayesian%20-%20Subject%20level%20(numpyro).ipynb)
| 3. Estimating meta-d' (single subject) using Bayesian modelling - pymc | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadPy/blob/master/notebooks/Example%202%20-%20Fitting%20Bayesian%20-%20Subject%20level%20(pymc).ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/embodied-computation-group/metadPy/blob/master/notebooks/Example%202%20-%20Fitting%20Bayesian%20-%20Subject%20level%20(pymc).ipynb)
| 4. Estimating meta-d' (group level) using Bayesian modelling - Numpyro | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadPy/blob/master/notebooks/Example%204%20-%20Fitting%20Bayesian%20-%20Group%20level%20(numpyro).ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/embodied-computation-group/metadPy/blob/master/notebooks/Example%202%20-%20Fitting%20Bayesian%20-%20Group%20level%20(numpyro).ipynb)


## Tutorials

| Notebook | Colab | nbViewer |
| --- | ---| --- |
| 1. What metacognition looks like? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadPy/blob/master/notebooks/1%20-%20What%20metacognition%20looks%20like.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/embodied-computation-group/metadPy/blob/master/notebooks/1%20-%20What%20metacognition%20looks%20like.ipynb)
| 2. Fitting the model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadPy/blob/master/notebooks/2%20-%20Fitting%20the%20model.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/embodied-computation-group/metadPy/blob/master/notebooks/2%20-%20Fitting%20the%20model.ipynb)
| 3. Comparison with the HMeta-d toolbox | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/embodied-computation-group/metadPy/blob/master/notebooks/3-Comparison%20with%20the%20hmeta-d%20toolbox.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/embodied-computation-group/metadPy/blob/master/notebooks/3-Comparison%20with%20the%20hmeta-d%20toolbox.ipynb)


# Importing data

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


```python
from metadPy.utils import responseSimulation

simulation = responseSimulation(d=1, metad=2.0, c=0, nRatings=4, nTrials=5000)
simulation.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Stimuli</th>
      <th>Responses</th>
      <th>Accuracy</th>
      <th>Confidence</th>
      <th>nTrial</th>
      <th>Subject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
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




    (array([640, 398, 405, 286, 455, 220,  62,  34]),
     array([ 34,  78, 225, 434, 310, 418, 398, 603]))



## Data visualization

You can easily visualize metacognition results using one of the plotting functions. Here, we will use the `plot_confidence` and the `plot_roc` functions to visualize the metacognitive performance of our participant.


```python
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from metadPy.plotting import plot_confidence, plot_roc

sns.set_context("talk")
```


```python
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
plot_confidence(nR_S1, nR_S2, ax=axs[0])
plot_roc(nR_S1, nR_S2, ax=axs[1])
sns.despine()
```


    
![png](./images/confidence_ROCAUC.png)
    


# Signal detection theory (SDT)

All metadPy functions are registred as Pandas flavors (see [pandas-flavor](https://pypi.org/project/pandas-flavor/)), which means that the functions can be called as a method from the result data frame. When using the default columns names (`Stimuli`, `Response`, `Accuracy`, `Confidence`), this significantly reduces the length of the function call, making your code more clean and readable.


```python
simulation.criterion()
```




    -0.0




```python
simulation.dprime()
```




    1.0007814013683562




```python
simulation.rates()
```




    (0.6916, 0.3084)




```python
simulation.roc_auc(nRatings=4)
```




    0.7817388829972878




```python
simulation.scores()
```




    (1729, 771, 771, 1729)



# Estimating meta dprime using Maximum Likelyhood Estimates (MLE)



```python
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

    meta-d' = 0    2.002298
    Name: meta_d, dtype: float64


# Estimating meta-dprime using hierarchical Bayesian modeling


## Subject level


```python
from metadPy.bayesian import hmetad
```


```python
model, trace = hmetad(
    data=simulation,
    nRatings=4,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
    backend="pymc"
)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [c1, d1, meta_d, cS1_hn, cS2_hn]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 00:06<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 10 seconds.



```python
az.plot_trace(trace, var_names=["meta_d", "cS2", "cS1"]);
```


    
![png](./images/subjectLevel.png)
    



```python
az.summary(trace, var_names=["meta_d", "cS2", "cS1"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>meta_d</th>
      <td>1.995</td>
      <td>0.055</td>
      <td>1.892</td>
      <td>2.099</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3151.0</td>
      <td>2932.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>cS2[0]</th>
      <td>0.508</td>
      <td>0.020</td>
      <td>0.472</td>
      <td>0.547</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>4170.0</td>
      <td>3143.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>cS2[1]</th>
      <td>1.041</td>
      <td>0.024</td>
      <td>0.991</td>
      <td>1.082</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3510.0</td>
      <td>3319.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>cS2[2]</th>
      <td>1.533</td>
      <td>0.029</td>
      <td>1.477</td>
      <td>1.586</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>3338.0</td>
      <td>3075.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>cS1[0]</th>
      <td>-1.487</td>
      <td>0.029</td>
      <td>-1.543</td>
      <td>-1.435</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3823.0</td>
      <td>3451.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>cS1[1]</th>
      <td>-0.989</td>
      <td>0.024</td>
      <td>-1.034</td>
      <td>-0.943</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3666.0</td>
      <td>3210.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>cS1[2]</th>
      <td>-0.474</td>
      <td>0.021</td>
      <td>-0.513</td>
      <td>-0.437</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3793.0</td>
      <td>2934.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Group level


```python
simulation = responseSimulation(
    d=1, metad=1.5, c=0, nRatings=4, nTrials=200, nSubjects=10
)
```


```python
model, trace = hmetad(
    data=simulation,
    nRatings=4,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
    subject="Subject",
    backend="numpyro"
)
```

    /opt/anaconda3/lib/python3.8/site-packages/metadPy/bayesian.py:357: UserWarning: There are not enough devices to run parallel chains: expected 4 but got 1. Chains will be drawn sequentially. If you are running MCMC in CPU, consider using `numpyro.set_host_device_count(4)` at the beginning of your program. You can double-check how many devices are available in your system using `jax.local_device_count()`.
      mcmc = MCMC(
      0%|                                                                                                                                                                                                                                                                          | 0/2000 [00:00<?, ?it/s]/opt/anaconda3/lib/python3.8/site-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
      warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
    sample: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:09<00:00, 218.90it/s, 31 steps of size 1.47e-01. acc. prob=0.88]
      0%|                                                                                                                                                                                                                                                                          | 0/2000 [00:00<?, ?it/s]/opt/anaconda3/lib/python3.8/site-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
      warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
    sample: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:09<00:00, 215.57it/s, 31 steps of size 1.52e-01. acc. prob=0.86]
      0%|                                                                                                                                                                                                                                                                          | 0/2000 [00:00<?, ?it/s]/opt/anaconda3/lib/python3.8/site-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
      warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
    sample: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:09<00:00, 220.97it/s, 31 steps of size 1.40e-01. acc. prob=0.90]
      0%|                                                                                                                                                                                                                                                                          | 0/2000 [00:00<?, ?it/s]/opt/anaconda3/lib/python3.8/site-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
      warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
    sample: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [00:09<00:00, 218.75it/s, 31 steps of size 1.31e-01. acc. prob=0.92]
    /opt/anaconda3/lib/python3.8/site-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
      warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '



```python
az.plot_posterior(trace, var_names=["mu_meta_d"], kind="hist", bins=20)
```




    <AxesSubplot:title={'center':'mu_meta_d'}>




    
![png](./images/groupLevel.png)
    


# References

[1] Maniscalco, B., & Lau, H. (2014). Signal Detection Theory Analysis of Type 1 and Type 2 Data: Meta-d′, Response-Specific Meta-d′, and the Unequal Variance SDT Model. In The Cognitive Neuroscience of Metacognition (pp. 25–66). Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-642-45190-4_3 

[2] Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. Frontiers in Human Neuroscience, 8. https://doi.org/10.3389/fnhum.2014.00443

[3] Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation of metacognitive efficiency from confidence ratings, Neuroscience of Consciousness, 3(1) nix007, https://doi.org/10.1093/nc/nix007


# Watermark


```python
%load_ext watermark
%watermark -n -u -v -iv -w -p metadPy,pymc
```

    Last updated: Wed Jun 29 2022
    
    Python implementation: CPython
    Python version       : 3.8.8
    IPython version      : 8.3.0
    
    metadPy: 0.0.1
    pymc   : 4.0.1
    
    matplotlib: 3.4.3
    arviz     : 0.12.1
    seaborn   : 0.11.2
    
    Watermark: 2.3.1
    

