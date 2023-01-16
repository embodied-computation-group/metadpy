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

(tutorial_1)=
# What metacognition looks like?
Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>  
Adapted from the tutorial proposed by the HMeta-d toolbox: https://github.com/metacoglab/HMeta-d/tree/master/CPC_metacog_tutorial

```{code-cell} ipython3
import matplotlib.pyplot as plt
import seaborn as sns
from metadpy.plotting import plot_confidence
from metadpy.utils import type2_SDT_simuation

sns.set_context("talk")
```

## Simulating ratings

+++

First, we will simulate data with high and low confidence noise, which should lead to higher and lower metacognition. This added noise represents a potential loss of information between type 1 task discrimination and 'metacognitive' information, i.e. type 2 confidence scores. We will then plot what the confidence scores would look like for each of these two scenarios (high and low confidence noise).

```{code-cell} ipython3
# Set up the parameters:
d = 2  # Set task performance (d')
c = 0  # Set task bias (c)
nTrials = 1000  # Set the number of trials performed
nRatings = 4  # Choose the rating scale to use
lowNoise, highNoise = 0, 0.7  # Set values for high and low confidence noise

# Simulate the responses for low and high confidence noise:
lowNoise_nR_S1, lowNoise_nR_S2 = type2_SDT_simuation(
    d=d, noise=lowNoise, c=c, nRatings=nRatings, nTrials=nTrials
)
highNoise_nR_S1, highNoise_nR_S2 = type2_SDT_simuation(
    d=d, noise=highNoise, c=c, nRatings=nRatings, nTrials=nTrials
)
```

Plot the confidence results for the two simulations:

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
plot_confidence(nR_S1=highNoise_nR_S1, nR_S2=highNoise_nR_S2, ax=axs[0])
axs[0].set_title("High noise")
plot_confidence(nR_S1=lowNoise_nR_S1, nR_S2=lowNoise_nR_S2, ax=axs[1])
axs[1].set_title("Low noise")
sns.despine()
```

**EXPLANATION:** You can see in the figure that the low confidence noise likely results in less overlap between the confidence distributions for correct and incorrect responses. It is actually the difference between these two distributions (one for correct responses, one for incorrect responses) that tells us how good metacognition is --> If you have higher confidence scores when you are correct and lower confidence scores when you are incorrect (and not much mixing between the two), the better metacognition will be. Another way of thinking about it is that the more 'noise' that is added as information passes from type 1 performance (e.g. correct vs incorrect perceptions) to type 2 performance (e.g. confidence in the decision that has been made), the less 'mixing' will occur between these two distributions and the larger (or 'better') metacognition will be.

+++

```{admonition} Exercise 1
Run the simulation and plot again a few times. Do you get the same results every time? Why / why not?
```

+++

## How does performance change metacognition?

+++

It is important to note that looking at the difference between the confidence distributions is an 'absolute' form of metacognition, which means that it will be affected by task performance (measured by d'). For example, if d' is higher (and the task is easier), there is more type 1 information on which to base metacognitive decisions than if d' is lower. We will explore this here by comparing the confidence scores with higher and lower d'.

+++

**Set up the parameters**

```{code-cell} ipython3
dHigh = 2  # Set task performance to be higher, with a larger d' value
dLow = 1  # Set task performance to be lower, with a smaller d' value
c = 0  # Set task bias (c)
nTrials = 1000  # Set the number of trials performed
nRatings = 4  # Choose the rating scale to use
noise = 0.2  # Set a value for the noise
```

**Simulate the responses for low and high task performance**

```{code-cell} ipython3
dHigh_nR_S1, dHigh_nR_S2 = type2_SDT_simuation(
    d=dHigh, noise=noise, c=c, nRatings=nRatings, nTrials=nTrials
)
dLow_nR_S1, dLow_nR_S2 = type2_SDT_simuation(
    d=dLow, noise=noise, c=c, nRatings=nRatings, nTrials=nTrials
)
```

**Plot the confidence results for the two simulations**

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
plot_confidence(nR_S1=dHigh_nR_S1, nR_S2=dHigh_nR_S2, ax=axs[0])
axs[0].set_title("High task performace")
plot_confidence(nR_S1=dLow_nR_S1, nR_S2=dLow_nR_S2, ax=axs[1])
axs[1].set_title("Low task performace")
sns.despine()
```

```{tip}
You can see in this figure that a higher task performance (d') results in a bigger difference between the confidence distributions, despite the same amount of confidence noise. Later on we will look at how we can correct for the level of d' to get a 'relative' measure of metacognition.
```

+++

```{admonition} Exercise 2
Change any of the parameters above and re-run this section to get an idea as to how each parameter may change the observed confidence scores.
```

```{code-cell} ipython3

```
