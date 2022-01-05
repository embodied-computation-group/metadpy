[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/LegrandNico/metadPy/blob/master/LICENSE) [![codecov](https://codecov.io/gh/LegrandNico/metadPy/branch/master/graph/badge.svg)](https://codecov.io/gh/LegrandNico/metadPy) [![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

***

<img src="https://github.com/LegrandNico/metadPy/raw/master/images/logo.png" align="left" alt="metadPy" height="250" HSPACE=30>

**metadPy** is an open-source Python package for cognitive modelling of behavioural data with a focus on metacognition. It is aimed to provide simple yet powerful functions to compute standard index and metric of signal detection theory (SDT) and metacognitive efficiency (meta-d’ and hierarchical meta-d’) [**1**, **2**, **3**]. The only input required is a data frame encoding task performances and confidence ratings at the trial level.

**metadPy** is written in Python 3. It uses [Numpy](https://numpy.org/), [Scipy](https://www.scipy.org/) and [Pandas](https://pandas.pydata.org/>) for most of its operation, comprizing meta-d’ estimation using maximum likelihood estimation (MLE). The (Hierarchical) Bayesian modelling of meta-d’ and m-ratio [**4**] is based on [JAX](https://jax.readthedocs.io/en/latest/) and [Numpyro](https://num.pyro.ai/en/latest/index.html#). Single subject modelling is also possible with [PyMC3](https://docs.pymc.io/>).

# Installation

The package can be installed using pip:

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

* [PyMC3](https://docs.pymc.io/>) (>=3.10.0) - only support non hierarchical modelling.

# Why metadPy?

metadPy stands for meta-d' (meta-d prime) in Python. meta-d' is a behavioural metric commonly used in consciousness and metacognition research. It is modelled to reflect metacognitive efficiency (i.e the relationship between subjective reports about performances and objective behaviour).

metadPy first aims to be the Python equivalent of the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) (Matlab and R). It tries to make these models available to a broader open-source ecosystem and to ease their use via cloud computing interfaces. One notable difference is that While the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) relies on JAGS for the Bayesian modelling of confidence data (see [**4**]) to analyse task performance and confidence ratings, metadPy is based on [JAX](https://jax.readthedocs.io/en/latest/) and [Numpyro](https://num.pyro.ai/en/latest/index.html#), which can easily be parallelized, flexibly uses CPU, GPU or TPU and offers a broader variety of MCMC sampling algorithms (comprising NUTS).

For an extensive introduction to metadPy, you can navigate the following notebooks that are Python adaptations of the introduction to the [hMeta-d toolbox](https://github.com/metacoglab/HMeta-d) written in Matlab by Olivia Faul for the [Zurich Computational Psychiatry course](https://github.com/metacoglab/HMeta-d/tree/master/CPC_metacog_tutorial).

## Examples 

| Notebook | Colab | nbViewer |
| --- | ---| --- |
| 1. Estimating meta-d' using MLE (subject and group level) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/metadPy/blob/master/notebooks/Example%201%20-%20Fitting%20MLE%20-%20Subject%20and%20group%20level.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/metadPy/blob/master/notebooks/Example%201%20-%20Fitting%20MLE%20-%20Subject%20and%20group%20level.ipynb)
| 2. Estimating meta-d' (single subject) using Bayesian modelling - Numpyro | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/metadPy/blob/master/notebooks/Example%202%20-%20Fitting%20Bayesian%20-%20Subject%20level%20(numpyro).ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/metadPy/blob/master/notebooks/Example%202%20-%20Fitting%20Bayesian%20-%20Subject%20level%20(numpyro).ipynb)
| 3. Estimating meta-d' (single subject) using Bayesian modelling - PyMC3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/metadPy/blob/master/notebooks/Example%202%20-%20Fitting%20Bayesian%20-%20Subject%20level%20(pymc3).ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/metadPy/blob/master/notebooks/Example%202%20-%20Fitting%20Bayesian%20-%20Subject%20level%20(pymc3).ipynb)
| 4. Estimating meta-d' (group level) using Bayesian modelling - Numpyro | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/metadPy/blob/master/notebooks/Example%204%20-%20Fitting%20Bayesian%20-%20Group%20level%20(numpyro).ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/metadPy/blob/master/notebooks/Example%202%20-%20Fitting%20Bayesian%20-%20Group%20level%20(numpyro).ipynb)


## Tutorials

| Notebook | Colab | nbViewer |
| --- | ---| --- |
| 1. What metacognition looks like? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/metadPy/blob/master/notebooks/1%20-%20What%20metacognition%20looks%20like.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/metadPy/blob/master/notebooks/1%20-%20What%20metacognition%20looks%20like.ipynb)
| 2. Fitting the model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/metadPy/blob/master/notebooks/2%20-%20Fitting%20the%20model.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/metadPy/blob/master/notebooks/2%20-%20Fitting%20the%20model.ipynb)
| 3. Hierarchical Bayesian models of metacognition (in prep) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/metadPy/blob/master/notebooks/3%20-%20Hierarchical%20Bayesian%20models%20of%20metacognition.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/metadPy/blob/master/notebooks/4%20-%20Comparison%20with%20the%20hmeta-d%20toolbox.ipynb)
| 4. Comparison with the HMeta-d toolbox | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegrandNico/metadPy/blob/master/notebooks/4%20-%20Comparison%20with%20the%20hmeta-d%20toolbox.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/LegrandNico/metadPy/blob/master/notebooks/4%20-%20Comparison%20with%20the%20hmeta-d%20toolbox.ipynb)

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

Due to the logical dependence between the `Stimuli`, `Responses` and `Accuracy` columns, in practice only two of those columns are necessary, the third being deduced from the others. Most of the functions in `metadPy` will accept DataFrames containing only two of these columns, and will automatically infer the missing information. Similarly, as the metacognition models described here does not incorporate the temporal dimension, the trial number is optional. 

`metadPy` includes a simulation function that will let you create one such data frame for one or many participants and condition, controlling for a variety of parameters. Here, we will simulate 200 trials from  participant having `d=1` and `c=0` (task performances) and a `meta-d=1.5` (metacognitive sensibility). The confidence ratings were provided using a 1-to-4 rating scale.


```python
from metadPy.utils import responseSimulation
simulation = responseSimulation(d=1, metad=1.5, c=0, 
                                nRatings=4, nTrials=200)
simulation
```
<div>
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
      <th>Condition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>195</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>196</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>196</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>197</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>197</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>198</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>198</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>199</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>199</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 7 columns</p>
</div>

```python
from metadPy.utils import trials2counts
nR_S1, nR_S2 = trials2counts(
    data=simulation, stimuli="Stimuli", accuracy="Accuracy",
    confidence="Confidence", nRatings=4)
```

## Data visualization

You can easily visualize metacognition results using one of the plotting functions. Here, we will use the `plot_confidence` and the `plot_roc` functions to visualize the metacognitive performance of our participant.


```python
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from metadPy.plotting import plot_confidence, plot_roc
sns.set_context('talk')
```

```python
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
plot_confidence(nR_S1, nR_S2, ax=axs[0])
plot_roc(nR_S1, nR_S2, ax=axs[1])
sns.despine()
```

![png](./images/confidence_ROCAUC.png)

# Signal detection theory (SDT)

```python
from metadPy.sdt import criterion, dprime, rates, roc_auc, scores
```

All metadPy functions are registred as Pandas flavors (see [pandas-flavor](https://pypi.org/project/pandas-flavor/)), which means that the functions can be called as a method from the result data frame. When using the default columns names (`Stimuli`, `Response`, `Accuracy`, `Confidence`), this significantly reduces the length of the function call, making your code more clean and readable.

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
    0.5797055057618438

```python
simulation.scores()
```
    (69, 31, 31, 69)

# Estimating meta dprime using Maximum Likelyhood Estimates (MLE)

```python
from metadPy.mle import metad

metad = metad(data=simulation, nRatings=4, stimuli='Stimuli',
              accuracy='Accuracy', confidence='Confidence', verbose=0)
print(f'meta-d\' = {str(metad["meta_da"])}')
```
    meta-d' = 0.5223485447196857

# Estimating meta-dprime using hierarchical Bayesian modeling

## Subject level

```python
import pymc3 as pm
from metadPy.hierarchical import hmetad
```

```python
model, trace = hmetad(data=simulation, nRatings=4, stimuli='Stimuli',
                      accuracy='Accuracy', confidence='Confidence')
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Sequential sampling (2 chains in 1 job)
    NUTS: [cS2_hn, cS1_hn, metad, d1, c1]
    



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
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:07<00:00 Sampling chain 0, 1 divergences]
</div>





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
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:07<00:00 Sampling chain 1, 0 divergences]
</div>



    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 15 seconds.
    /usr/local/lib/python3.6/dist-packages/arviz/data/io_pymc3.py:314: UserWarning: Could not compute log_likelihood, it will be omitted. Check your model object or set log_likelihood=False
      warnings.warn(warn_msg)
    There was 1 divergence after tuning. Increase `target_accept` or reparameterize.
    There was 1 divergence after tuning. Increase `target_accept` or reparameterize.
    


```python
pm.traceplot(trace, var_names=['metad', 'cS2', 'cS1']);
```
 
![png](./images/hmetad.png)

```python
pm.summary(trace)
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
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>metad</th>
      <td>0.534</td>
      <td>0.245</td>
      <td>0.018</td>
      <td>0.960</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>1779.0</td>
      <td>1779.0</td>
      <td>1810.0</td>
      <td>1376.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>cS1[0]</th>
      <td>-1.488</td>
      <td>0.139</td>
      <td>-1.755</td>
      <td>-1.239</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>1871.0</td>
      <td>1846.0</td>
      <td>1879.0</td>
      <td>1615.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>cS1[1]</th>
      <td>-0.928</td>
      <td>0.109</td>
      <td>-1.125</td>
      <td>-0.725</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>2161.0</td>
      <td>2121.0</td>
      <td>2155.0</td>
      <td>1813.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>cS1[2]</th>
      <td>-0.429</td>
      <td>0.092</td>
      <td>-0.596</td>
      <td>-0.259</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>1987.0</td>
      <td>1909.0</td>
      <td>1988.0</td>
      <td>1742.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>cS2[0]</th>
      <td>0.486</td>
      <td>0.093</td>
      <td>0.317</td>
      <td>0.664</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2200.0</td>
      <td>2197.0</td>
      <td>2188.0</td>
      <td>1710.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>cS2[1]</th>
      <td>0.904</td>
      <td>0.106</td>
      <td>0.711</td>
      <td>1.103</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>2051.0</td>
      <td>2034.0</td>
      <td>2049.0</td>
      <td>1702.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>cS2[2]</th>
      <td>1.408</td>
      <td>0.131</td>
      <td>1.179</td>
      <td>1.663</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>1784.0</td>
      <td>1772.0</td>
      <td>1786.0</td>
      <td>1598.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>

## Group level

```python
simulation = responseSimulation(d=1, metad=1.5, c=0, nRatings=4,
                                nTrials=200, nSubjects=10)
```

```python
model, trace = hmetad(
    data=simulation, nRatings=4, stimuli='Stimuli', accuracy='Accuracy',
    confidence='Confidence', subject='Subject')
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Sequential sampling (2 chains in 1 job)
    NUTS: [cS2_hn, cS1_hn, epsilon_logMratio, delta_tilde, sigma_delta, mu_logMratio, d1_tilde, c1_tilde, sigma_d1, sigma_c2, sigma_c1, mu_d1, mu_c2, mu_c1]
    



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
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:45<00:00 Sampling chain 0, 13 divergences]
</div>





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
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:38<00:00 Sampling chain 1, 11 divergences]
</div>



    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 84 seconds.
    There were 13 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 24 divergences after tuning. Increase `target_accept` or reparameterize.
    The estimated number of effective samples is smaller than 200 for some parameters.
    


```python
az.plot_posterior(trace, var_names=['mu_logMratio'], kind='hist', bins=20)
```
![png](./images/groupLevel.png)


# References

[1] Maniscalco, B., & Lau, H. (2014). Signal Detection Theory Analysis of Type 1 and Type 2 Data: Meta-d′, Response-Specific Meta-d′, and the Unequal Variance SDT Model. In The Cognitive Neuroscience of Metacognition (pp. 25–66). Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-642-45190-4_3 

[2] Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach for estimating metacognitive sensitivity from confidence ratings. Consciousness and Cognition, 21(1), 422–430. doi:10.1016/j.concog.2011.09.021

[3] Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. Frontiers in Human Neuroscience, 8. https://doi.org/10.3389/fnhum.2014.00443

[4] Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation of metacognitive efficiency from confidence ratings, Neuroscience of Consciousness, 3(1) nix007, https://doi.org/10.1093/nc/nix007
