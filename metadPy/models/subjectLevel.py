# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
"""
This is an internal function. The subject level modeling shoul be called using
the metadPy.hierarchical.metad function instead.
"""

import numpy as np
import pymc3.distributions.transforms as tr
from pymc3 import (
    Binomial,
    Bound,
    Deterministic,
    Model,
    Multinomial,
    Normal,
    math,
    sample,
)


def cumulative_normal(x):
    """Cummulative normal distribution"""
    return 0.5 + 0.5 * math.erf(x / math.sqrt(2))


def hmetad_subjectLevel(
    data, chains=3, tune=1000, draws=1000, cores=1, sample_model=True
):
    """Compute hierachical meta-d' at the subject level.

    Parameters
    ----------
    data : dict
        Response data.
    chains : int
        The number of chains to sample. Defaults to `3`.
    tune : int
        Number of iterations to tune. Defaults to `1000`.
    draws : int
        The number of samples to draw. Defaults to `1000`.

    Returns
    -------
    model : :py:class:`pymc3.Model` instance
        The pymc3 model.
    traces : dict
        Dictionnary of the results and logs:
            * `'trace'`: the MCMC traces

    References
    ----------
    .. [#] Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation
    of metacognitive efficiency from confidence ratings, Neuroscience of
    Consciousness, 3(1) nix007, https://doi.org/10.1093/nc/nix007
    """
    nRatings = data["nratings"]
    with Model() as model:

        # Type 1 priors
        c1 = Normal("c1", mu=0.0, tau=2)
        d1 = Normal("d1", mu=0.0, tau=0.5)

        # TYPE 1 SDT BINOMIAL MODEL
        h = cumulative_normal(d1 / 2 - c1)
        f = cumulative_normal(-d1 / 2 - c1)
        H = Binomial("H", data["S"], h, observed=data["H"])
        FA = Binomial("FA", data["N"], f, observed=data["FA"])

        # Type 2 priors
        meta_d = Normal("metad", mu=d1, tau=2)

        # Specify ordered prior on criteria
        # bounded above and below by Type 1 c1
        BoundedNormal_cS1 = Bound(Normal, upper=data["c1"] - data["Tol"])
        cS1 = BoundedNormal_cS1(
            "cS1",
            mu=0.0,
            tau=2,
            shape=nRatings - 1,
            transform=tr.Ordered(),
            testval=np.arange(-(nRatings - 1), 0),
        )

        BoundedNormal_cS2 = Bound(Normal, lower=data["c1"] + data["Tol"])
        cS2 = BoundedNormal_cS2(
            "cS2",
            mu=0.0,
            tau=2,
            shape=nRatings - 1,
            transform=tr.Ordered(),
            testval=np.arange(1, nRatings),
        )

        # Means of SDT distributions
        S2mu = math.flatten(meta_d / 2, 1)
        S1mu = math.flatten(-meta_d / 2, 1)

        # Calculate normalisation constants
        C_area_rS1 = cumulative_normal(c1 - S1mu)
        I_area_rS1 = cumulative_normal(c1 - S2mu)
        C_area_rS2 = 1 - cumulative_normal(c1 - S2mu)
        I_area_rS2 = 1 - cumulative_normal(c1 - S1mu)

        # Get nC_rS1 probs
        nC_rS1 = cumulative_normal(cS1 - S1mu) / C_area_rS1
        nC_rS1 = Deterministic(
            "nC_rS1",
            math.concatenate(
                (
                    [
                        cumulative_normal(cS1[0] - S1mu) / C_area_rS1,
                        nC_rS1[1:] - nC_rS1[:-1],
                        (
                            (
                                cumulative_normal(c1 - S1mu)
                                - cumulative_normal(cS1[(nRatings - 2)] - S1mu)
                            )
                            / C_area_rS1
                        ),
                    ]
                ),
                axis=0,
            ),
        )

        # Get nI_rS2 probs
        nI_rS2 = (1 - cumulative_normal(cS2 - S1mu)) / I_area_rS2
        nI_rS2 = Deterministic(
            "nI_rS2",
            math.concatenate(
                (
                    [
                        (
                            (1 - cumulative_normal(c1 - S1mu))
                            - (1 - cumulative_normal(cS2[0] - S1mu))
                        )
                        / I_area_rS2,
                        nI_rS2[:-1]
                        - (1 - cumulative_normal(cS2[1:] - S1mu)) / I_area_rS2,
                        (1 - cumulative_normal(cS2[nRatings - 2] - S1mu)) / I_area_rS2,
                    ]
                ),
                axis=0,
            ),
        )

        # Get nI_rS1 probs
        nI_rS1 = (-cumulative_normal(cS1 - S2mu)) / I_area_rS1
        nI_rS1 = Deterministic(
            "nI_rS1",
            math.concatenate(
                (
                    [
                        cumulative_normal(cS1[0] - S2mu) / I_area_rS1,
                        nI_rS1[:-1] + (cumulative_normal(cS1[1:] - S2mu)) / I_area_rS1,
                        (
                            cumulative_normal(c1 - S2mu)
                            - cumulative_normal(cS1[(nRatings - 2)] - S2mu)
                        )
                        / I_area_rS1,
                    ]
                ),
                axis=0,
            ),
        )

        # Get nC_rS2 probs
        nC_rS2 = (1 - cumulative_normal(cS2 - S2mu)) / C_area_rS2
        nC_rS2 = Deterministic(
            "nC_rS2",
            math.concatenate(
                (
                    [
                        (
                            (1 - cumulative_normal(c1 - S2mu))
                            - (1 - cumulative_normal(cS2[0] - S2mu))
                        )
                        / C_area_rS2,
                        nC_rS2[:-1]
                        - ((1 - cumulative_normal(cS2[1:] - S2mu)) / C_area_rS2),
                        (1 - cumulative_normal(cS2[nRatings - 2] - S2mu)) / C_area_rS2,
                    ]
                ),
                axis=0,
            ),
        )

        # Avoid underflow of probabilities
        nC_rS1 = math.switch(nC_rS1 < data["Tol"], data["Tol"], nC_rS1)
        nI_rS2 = math.switch(nI_rS2 < data["Tol"], data["Tol"], nI_rS2)
        nI_rS1 = math.switch(nI_rS1 < data["Tol"], data["Tol"], nI_rS1)
        nC_rS2 = math.switch(nC_rS2 < data["Tol"], data["Tol"], nC_rS2)

        # TYPE 2 SDT MODEL (META-D)
        # Multinomial likelihood for response counts ordered as c(nR_S1,nR_S2)
        Multinomial(
            "CR_counts",
            data["CR"],
            nC_rS1,
            shape=nRatings,
            observed=data["counts"][:nRatings],
        )
        Multinomial(
            "FA_counts",
            FA,
            nI_rS2,
            shape=nRatings,
            observed=data["counts"][nRatings : nRatings * 2],
        )
        Multinomial(
            "M_counts",
            data["M"],
            nI_rS1,
            shape=nRatings,
            observed=data["counts"][nRatings * 2 : nRatings * 3],
        )
        Multinomial(
            "H_counts",
            H,
            nC_rS2,
            shape=nRatings,
            observed=data["counts"][nRatings * 3 : nRatings * 4],
        )

        if sample_model is True:
            trace = sample(
                draws,
                chains=chains,
                progressbar=True,
                trace=[meta_d, cS1, cS2],
                tune=tune,
                cores=cores,
            )

            return model, trace

        else:
            return model
