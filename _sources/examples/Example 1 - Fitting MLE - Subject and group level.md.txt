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

+++ {"id": "copyrighted-border"}

(example_1)=
# Fitting single subject data using MLE
Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

```{code-cell} ipython3
:id: unavailable-groove

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from metadpy import load_dataset
from metadpy.mle import metad
from metadpy.plotting import plot_confidence, plot_roc

sns.set_context("talk")
```

+++ {"id": "2oE_wkIxVPbe"}

In this notebook, we are going to estimate meta-*d'* using Maximum Likelihood Estimation ([MLE](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)) {cite:p}`fleming:2014,maniscalo:2014,maniscalo:2012` using the function implemented in [metadpy](https://github.com/LegrandNico/metadpy). This function is directly adapted from the transcription of the Matlab `fit_meta_d_MLE.m` by Alan Lee that can be retrieved [here](http://www.columbia.edu/~bsm2105/type2sdt/).

We are going to see, however, that [metadpy](https://github.com/LegrandNico/metadpy) greatly simplifies the preprocessing of raw data, letting the user fit the model for many participants/groups/conditions from the results data frame in a single command call. Another advantage here is that the python code supporting the model fitting is optimized using [Numba](http://numba.pydata.org/), which greatly improves its performance.

+++ {"id": "current-valuation"}

## From response-signal arrays

```{code-cell} ipython3
:id: controversial-executive

# Create responses data
nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 396
id: Q2dmDRLAlBaV
outputId: 023cb955-ea90-426a-d40b-3fe2090414ac
---
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
plot_confidence(nR_S1, nR_S2, ax=axs[0])
plot_roc(nR_S1, nR_S2, ax=axs[1])
sns.despine()
```

+++ {"id": "GJFs74YdcqxR"}

The model is fitted using the `metadpy.mle.metad()` function. This function accepts response-signal arrays as input if the data comes from a single subject.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: experienced-ottawa
outputId: 29f54c96-6016-4140-d435-9ae2a11d8c0d
---
output = metad(nR_S1=nR_S1, nR_S2=nR_S2)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 81
id: 5f8Sn_vVcnbw
outputId: 7c33fb1a-aa9d-4eae-cdc6-ae221d43d39d
---
output
```

+++ {"id": "Iwh5RzuddDSw"}

The function will return a data frame containng the `dprime`, `metad`, `m_ratio` and `m_diff` scores for this participant.

+++ {"id": "recognized-testament"}

## From a data frame
To simplify the preprocessing steps, the model can also be fitted directly from the raw result data frame. The data frame should contain the following columns:

* `Stimuli`: Which of the two stimuli was presented [0 or 1].
* `Response` or `Accuracy`: The response provided by the participant or the accuracy [0 or 1].
* `Confidence`: The confidence level [can be continuous or discrete].

In addition, it can also integrate:
* `Subject`: The subject ID.
* `within` or `between`: The condition or the group ID (if many groups or conditions were used).

Note that the MLE method will always fit the participant separately (i.e. in a non-hierarchical way), which means that the results will be the same by fitting each participant and condition separately (e.g. in a for loop).

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 206
id: cloudy-possession
outputId: 2c1fbdc9-6d24-40d5-e55e-8c4f52b7fa0d
---
df = load_dataset("rm")
df.head()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: fuzzy-minutes
outputId: c14ad137-d5c5-4350-a1e2-a51c4aad7594
---
subject_fit = metad(
    data=df[df.Subject == 0].copy(),
    nRatings=4,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
    padding=True,
)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 81
id: Dc8nUuskqoMs
outputId: 1b57274b-6b0a-4742-f38c-8981f8dd2031
---
subject_fit.head()
```

+++ {"id": "corporate-arbitration"}

# Fitting at the group level

+++ {"id": "cutting-carbon"}

## Using a dataframe

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: alien-vocabulary
outputId: 96a96850-e578-48df-c555-1dcbb33bdfe6
---
group_fit = metad(
    data=df,
    nRatings=4,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
    subject="Subject",
    padding=True,
    within="Condition",
)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 206
id: n2ISHkFTnzre
outputId: 288edc49-0844-4e67-b225-0774a0c1bfa2
---
group_fit.head()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 352
id: passive-entity
outputId: f0d76dee-523f-4ddc-88dc-db5ce0a3ced8
---
_, axs = plt.subplots(1, 4, figsize=(12, 5), sharex=True)

for i, metric in enumerate(["dprime", "meta_d", "m_ratio", "m_diff"]):

    sns.boxplot(data=group_fit, x="Condition", y=metric, ax=axs[i])
    sns.stripplot(data=group_fit, x="Condition", y=metric, color="k", ax=axs[i])

plt.tight_layout()
sns.despine()
```

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p metadpy,pytensor,pymc
```

```{code-cell} ipython3

```
