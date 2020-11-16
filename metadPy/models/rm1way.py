# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>
"""
This is an internal function. The subject level modeling shoul be called using
the metadPy.hierarchical.metad function instead.
"""

import numpy as np
import theano.tensor as tt
from pymc3 import (
    Model,
    Normal,
    Binomial,
    Gamma,
    Multinomial,
    Bound,
    Deterministic,
    math,
    sample,
    Beta,
)


def cumulative_normal(x):
    """Cummulative normal distribution"""
    return 0.5 + 0.5 * math.erf(x / math.sqrt(2))


def hmetad_rm1way(data, chains=3, tune=1000, draws=1000):
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
    traces : dict
        Dictionnary of the results and logs:
            * `'trace'`: the MCMC traces

    References
    ----------
    .. [#] Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation
    of metacognitive efficiency from confidence ratings, Neuroscience of
    Consciousness, 3(1) nix007, https://doi.org/10.1093/nc/nix007
    """
    nSubj = data["nSubj"]
    subID = np.arange(nSubj, dtype="int")

    nCond = len(data["condition"].unique)
    cond = data["condition"]

    hits = data["hits"]
    falsealarms = data["falsealarms"]
    s = data["s"]
    n = data["n"]
    counts = data["counts"]
    nRatings = data["nRatings"]
    Tol = data["Tol"]
    cr = data["cr"]
    m = data["m"]

    with Model():

        # Hyperpriors
        mu_c2 = Normal("mu_c2", mu=0.0, tau=0.01, shape=1)
        sigma_c2 = Bound(Normal, lower=0.0)("sigma_c2", mu=0, tau=0.01)
        lambda_c2 = Deterministic("lambda_c2", sigma_c2 ** -2)

        mu_D = Normal("mu_D", mu=0.0, tau=0.01, shape=1)
        sigma_D = Bound(Normal, lower=0.0)("sigma_D", mu=0, tau=0.01)
        lambda_D = Deterministic("lambda_D", sigma_D ** -2)
        sigD = Deterministic("sigD", 1 / math.sqrt(lambda_D))

        mu_Cond1 = Normal("mu_Cond1", mu=0.0, tau=0.01, shape=1)
        sigma_Cond1 = Bound(Normal, lower=0.0)("sigma_Cond1", mu=0, tau=0.01)
        lambda_Cond1 = Deterministic("lambda_D", sigma_Cond1 ** -2)
        sigCond1 = Deterministic("sigCond1", 1 / math.sqrt(lambda_Cond1))

        # Hyperpriors - Subject level
        dbase = Normal("dbase", mu=mu_D, tau=lambda_D, shape=nSubj)
        Bd_Cond1 = Normal("Bd_Cond1", mu=mu_Cond1, tau=lambda_Cond1, shape=nSubj)
        tau = Gamma("tau", alpha=0.0, beta=0.01, shape=nSubj)

        # Hypterprior - Condition level
        mu_regression = Deterministic(
            "mu_regression", dbase + Bd_Cond1 * cond, shape=(1, nSubj, nCond)
        )
        logMratio = Normal("logMratio", mu_regression, tau)
        Mratio = Deterministic("mRatio", math.exp(logMratio), shape=(1, nSubj, nCond))

        # Means of SDT distributions
        mu = Deterministic("mu", mRatio * d1, shape=(1, nSubj, nCond))
        S2mu = Deterministic("S2mu", mu / 2, shape=(1, nSubj, nCond))
        S1mu = Deterministic("S2mu", -mu / 2, shape=(1, nSubj, nCond))

        # Type 1 priors
        # c1 = Normal("c1", mu=0.0, tau=2, shape=1)
        # d1 = Normal("d1", mu=0.0, tau=0.5, shape=1)

        # TYPE 1 SDT BINOMIAL MODEL
        h = cumulative_normal(d1 / 2 - c1)
        f = cumulative_normal(-d1 / 2 - c1)
        H = Binomial("H", s[subID, cond], h[0, subID, cond], observed=hits[subID, cond])
        FA = Binomial(
            "FA", n[subID, cond], f[0, subID, cond], observed=falsealarms[subID, cond]
        )

        # Specify ordered prior on criteria
        # bounded above and below by Type 1 c1
        cS1 = Deterministic(
            "cS1",
            tt.sort(
                Bound(Normal, upper=c1 - Tol)(
                    "cS1_raw",
                    mu=mu_c2,
                    tau=lambda_c2,
                    shape=(nRatings - 1, nSubj, nCond),
                )
            ),
        )
        cS2 = Deterministic(
            "cS2",
            tt.sort(
                Bound(Normal, lower=c1 + Tol)(
                    "cS2_raw",
                    mu=mu_c2,
                    tau=lambda_c2,
                    shape=(nRatings - 1, nSubj, nCond),
                )
            ),
        )

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
                        nC_rS1[1:, subID, cond] - nC_rS1[:-1, subID, cond],
                        (
                            (
                                cumulative_normal(c1 - S1mu)
                                - cumulative_normal(
                                    cS1[(nRatings - 2, subID, cond)] - S1mu
                                )
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
                            - (1 - cumulative_normal(cS2[0, subID, cond] - S1mu))
                        )
                        / I_area_rS2,
                        nI_rS2[:-1, subID, cond]
                        - (1 - cumulative_normal(cS2[1:, subID, cond] - S1mu))
                        / I_area_rS2,
                        (1 - cumulative_normal(cS2[nRatings - 2, subID, cond] - S1mu))
                        / I_area_rS2,
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
                        cumulative_normal(cS1[0, subID, cond] - S2mu) / I_area_rS1,
                        nI_rS1[:-1]
                        + (cumulative_normal(cS1[1:, subID, cond] - S2mu)) / I_area_rS1,
                        (
                            cumulative_normal(c1 - S2mu)
                            - cumulative_normal(cS1[nRatings - 2, subID, cond] - S2mu)
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
                            - (1 - cumulative_normal(cS2[0, subID, cond] - S2mu))
                        )
                        / C_area_rS2,
                        nC_rS2[:-1]
                        - (
                            (1 - cumulative_normal(cS2[1:, subID, cond] - S2mu))
                            / C_area_rS2
                        ),
                        (1 - cumulative_normal(cS2[nRatings - 2, subID, cond] - S2mu))
                        / C_area_rS2,
                    ]
                ),
                axis=0,
            ),
        )

        # Avoid underflow of probabilities
        nC_rS1 = math.switch(nC_rS1 < Tol, Tol, nC_rS1)
        nI_rS2 = math.switch(nI_rS2 < Tol, Tol, nI_rS2)
        nI_rS1 = math.switch(nI_rS1 < Tol, Tol, nI_rS1)
        nC_rS2 = math.switch(nC_rS2 < Tol, Tol, nC_rS2)

        # TYPE 2 SDT MODEL (META-D)
        # Multinomial likelihood for response counts ordered as c(nR_S1,nR_S2)
        Multinomial(
            "CR_counts",
            cr,
            nC_rS1.swapaxes(axis1=1, axis2=0),
            shape=(nSubj, nCond, nRatings),
            observed=counts[subID, cond, :nRatings],
        )
        Multinomial(
            "FA_counts",
            FA,
            nI_rS2.swapaxes(axis1=1, axis2=0),
            shape=(nSubj, nCond, nRatings),
            observed=counts[subID, cond, nRatings : nRatings * 2],
        )
        Multinomial(
            "M_counts",
            m,
            nI_rS1.swapaxes(axis1=1, axis2=0),
            shape=(nSubj, nCond, nRatings),
            observed=counts[subID, cond, nRatings * 2 : nRatings * 3],
        )
        Multinomial(
            "H_counts",
            H,
            nC_rS2.swapaxes(axis1=1, axis2=0),
            shape=(nSubj, nCond, nRatings),
            observed=counts[subID, cond, nRatings * 3 : nRatings * 4],
        )

        trace = sample(
            progressbar=True,
            trace=[sigma_logMratio, meta_d, mRatio, mu_logMratio, mu_d1, mu_c],
        )

    return trace
