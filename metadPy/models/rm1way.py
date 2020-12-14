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
    HalfNormal,
    Gamma,
    Multinomial,
    Bound,
    Deterministic,
    math,
    sample,
    HalfCauchy,
)


def cumulative_normal(x):
    """Cummulative normal distribution"""
    return 0.5 + 0.5 * math.erf(x / math.sqrt(2))


def hmetad_rm1way(data, sample_model=True, **kwargs):
    """Compute hierachical meta-d' at the subject level.

    Parameters
    ----------
    data : dict
        Response data.
    sample_model : boolean
        If `False`, only the model is returned without sampling.
    **kwargs : keyword arguments
        All keyword arguments are passed to `func::pymc3.sampling.sample`.

    Returns
    -------
    model : :py:class:`pymc3.Model` instance
        The pymc3 model. Encapsulates the variables and likelihood factors.
    trace : :py:class:`pymc3.backends.base.MultiTrace` or
        :py:class:`arviz.InferenceData`
        A `MultiTrace` or `ArviZ InferenceData` object that contains the
        samples.

    References
    ----------
    .. [#] Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation
    of metacognitive efficiency from confidence ratings, Neuroscience of
    Consciousness, 3(1) nix007, https://doi.org/10.1093/nc/nix007
    """
    nSubj = data["nSubj"]
    nCond = data["nCond"]
    cond = data["condition"]
    nRatings = data["nRatings"]
    hits = data["hits"].reshape(nSubj, nCond, 1).repeat(nRatings, axis=2)
    falsealarms = data["falsealarms"].reshape(nSubj, nCond, 1).repeat(nRatings, axis=2)
    s = data["s"]
    n = data["n"]
    counts = data["counts"]
    Tol = data["Tol"]
    cr = data["cr"].reshape(nSubj, nCond, 1).repeat(nRatings, axis=2)
    m = data["m"].reshape(nSubj, nCond, 1).repeat(nRatings, axis=2)
    c1 = data["c1"]
    d1 = data["d1"]

    with Model() as model:

        # Hyperpriors
        mu_c2 = Normal("mu_c2", mu=0.0, tau=0.01, shape=(nRatings - 1, nSubj, nCond))
        sigma_c2 = Bound(Normal, lower=0.0)(
            "sigma_c2", mu=0, tau=0.01, shape=(nRatings - 1, nSubj, nCond)
        )
        lambda_c2 = Deterministic("lambda_c2", sigma_c2 ** -2)

        mu_D = Normal("mu_D", mu=0.0, tau=0.01, testval=0)
        sigma_D = HalfCauchy("sigma_D", beta=5)

        mu_Cond1 = Normal("mu_Cond1", mu=0.0, tau=0.01, testval=0)
        sigma_Cond1 = HalfCauchy("sigma_Cond1", beta=5)

        #############################
        # Hyperpriors - Subject level
        #############################
        dbase_tilde = Normal("dbase_tilde", mu=0, sigma=1, shape=(1, nSubj, 1))
        dbase = Deterministic("dbase", mu_D + sigma_D * dbase_tilde)

        Bd_Cond1_tilde = Normal("Bd_Cond1_tilde", mu=0, sigma=1, shape=(1, nSubj, 1))
        Bd_Cond1 = Deterministic("Bd_Cond1", mu_Cond1 + sigma_Cond1 * Bd_Cond1_tilde)

        tau = Gamma("tau", alpha=0.01, beta=0.01, shape=(1, nSubj, 1))

        ###############################
        # Hypterprior - Condition level
        ###############################
        mu_regression = Deterministic("mu_regression", dbase + Bd_Cond1 * cond)
        logMratio = Normal("logMratio", mu_regression, tau=tau, shape=(1, nSubj, nCond))
        mRatio = Deterministic("mRatio", math.exp(logMratio))

        # Means of SDT distributions
        metad = Deterministic("metad", mRatio * d1)
        S2mu = Deterministic("S2mu", metad / 2)
        S1mu = Deterministic("S1mu", -metad / 2)

        # Specify ordered prior on criteria
        # bounded above and below by Type 1 c1
        cS1_hn = HalfNormal(
            "cS1_hn",
            tau=lambda_c2,
            shape=(nRatings - 1, nSubj, nCond),
            testval=np.linspace(1.5, 0.5, nRatings - 1)
            .reshape(nRatings - 1, 1, 1)
            .repeat(nSubj, axis=1)
            .repeat(nCond, axis=2),
        )
        cS1 = Deterministic("cS1", mu_c2 - cS1_hn + (c1 - data["Tol"]))

        cS2_hn = HalfNormal(
            "cS2_hn",
            tau=lambda_c2,
            shape=(nRatings - 1, nSubj, nCond),
            testval=np.linspace(1.5, 0.5, nRatings - 1)
            .reshape(nRatings - 1, 1, 1)
            .repeat(nSubj, axis=1)
            .repeat(nCond, axis=2),
        )
        cS2 = Deterministic("cS2", mu_c2 + cS2_hn + (c1 - data["Tol"]))

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
                            - cumulative_normal(cS1[nRatings - 2] - S2mu)
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
        nC_rS1 = math.switch(nC_rS1 < Tol, Tol, nC_rS1).transpose(1, 2, 0)
        nI_rS2 = math.switch(nI_rS2 < Tol, Tol, nI_rS2).transpose(1, 2, 0)
        nI_rS1 = math.switch(nI_rS1 < Tol, Tol, nI_rS1).transpose(1, 2, 0)
        nC_rS2 = math.switch(nC_rS2 < Tol, Tol, nC_rS2).transpose(1, 2, 0)

        # TYPE 2 SDT MODEL (META-D)
        # Multinomial likelihood for response counts
        Multinomial(
            "CR_counts",
            cr,
            nC_rS1,
            # shape=(nRatings, nSubj, nCond),
            observed=counts[:, :, :nRatings],
        )
        Multinomial(
            "FA_counts",
            falsealarms,
            nI_rS2,
            # shape=(nRatings, nSubj, nCond),
            observed=counts[:, :, nRatings : nRatings * 2],
        )
        Multinomial(
            "M_counts",
            m,
            nI_rS1,
            # shape=(nRatings, nSubj, nCond),
            observed=counts[:, :, nRatings * 2 : nRatings * 3],
        )
        Multinomial(
            "H_counts",
            hits,
            nC_rS2,
            # shape=(nRatings, nSubj, nCond),
            observed=counts[:, :, nRatings * 3 : nRatings * 4],
        )

        if sample_model is True:

            trace = sample(
                progressbar=True,
                trace=[
                    mRatio,
                    mu_c2,
                    sigma_c2,
                    lambda_c2,
                    mu_D,
                    sigma_D,
                    mu_Cond1,
                    sigma_Cond1,
                    metad,
                    tau,
                    dbase,
                    Bd_Cond1,
                ],
                **kwargs,
            )

            return model, trace

        else:
            return model
