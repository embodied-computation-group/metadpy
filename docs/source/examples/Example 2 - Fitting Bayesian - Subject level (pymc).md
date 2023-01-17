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

+++ {"id": "regulated-swiss"}

(example_2)=
# Fitting single subject data using Bayesian estimation
Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

```{code-cell} ipython3
:id: relevant-market

import arviz as az
import numpy as np
from metadpy.bayesian import hmetad
```

+++ {"id": "operating-aerospace"}

## From response-signal arrays

```{code-cell} ipython3
:id: worldwide-utility

# Create responses data
nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
```

+++ {"id": "QBbR-PBdsMpH"}

This function will return two variable. The first one is a pymc model variable

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 186
id: dried-sport
outputId: 378937a2-2931-4702-a668-21e4193c30f5
---
model, traces = hmetad(nR_S1=nR_S1, nR_S2=nR_S2)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 457
id: ZQrA4ZR0rtkg
outputId: a6cf2c8e-9e47-4314-eeac-90b8636b5d05
---
az.plot_trace(traces, var_names=["c1", "d1", "meta_d", "cS1", "cS2"]);
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 269
id: YS-BtDxer1-Q
outputId: 174b4bec-f2a1-4f33-e7e4-fbc91924f1b3
---
az.summary(traces, var_names=["c1", "d1", "meta_d", "cS1", "cS2"])
```

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p metadpy,pytensor,pymc
```

```{code-cell} ipython3

```
