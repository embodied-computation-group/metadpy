# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import theano.tensor as tt
from pymc3 import (
    Beta,
    Binomial,
    Deterministic,
    HalfNormal,
    Model,
    Multinomial,
    Normal,
    math,
    sample,
)


def cumulative_normal(x):
    """Cummulative normal distribution"""
    return 0.5 + 0.5 * math.erf(x / math.sqrt(2))


def hmetad_groupLevel(data: dict, sample_model: bool = True, **kwargs):
    """Compute hierachical meta-d' at the subject level.

    This is an internal function. The group level model must be
    called using :py:func:`metadPy.hierarchical.hmetad`.

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
    hits = data["hits"]
    falsealarms = data["falsealarms"]
    s = data["s"]
    n = data["n"]
    counts = data["counts"]
    nRatings = data["nRatings"]
    Tol = data["Tol"]
    cr = data["cr"]
    m = data["m"]

    with Model() as model:

        # hyperpriors on d, c and c2
        mu_c1 = Normal(
            "mu_c1", mu=0, tau=0.01, shape=(1), testval=np.random.rand() * 0.1
        )
        mu_c2 = Normal(
            "mu_c2", mu=0, tau=0.01, shape=(1, 1), testval=np.random.rand() * 0.1
        )
        mu_d1 = Normal(
            "mu_d1", mu=0, tau=0.01, shape=(1), testval=np.random.rand() * 0.1
        )

        sigma_c1 = HalfNormal(
            "sigma_c1", tau=0.01, shape=(1), testval=np.random.rand() * 0.1
        )
        sigma_c2 = HalfNormal(
            "sigma_c2", tau=0.01, shape=(1, 1), testval=np.random.rand() * 0.1
        )
        sigma_d1 = HalfNormal(
            "sigma_d1", tau=0.01, shape=(1), testval=np.random.rand() * 0.1
        )

        # Type 1 priors
        c1_tilde = Normal("c1_tilde", mu=0, sigma=1, shape=(nSubj, 1))
        c1 = Deterministic("c1", mu_c1 + sigma_c1 * c1_tilde)

        d1_tilde = Normal("d1_tilde", mu=0, sigma=1, shape=(nSubj, 1))
        d1 = Deterministic("d1", mu_d1 + sigma_d1 * d1_tilde)

        # TYPE 1 SDT BINOMIAL MODEL
        h = cumulative_normal(d1 / 2 - c1)
        f = cumulative_normal(-d1 / 2 - c1)
        H = Binomial("H", n=s, p=h, observed=hits)
        FA = Binomial("FA", n=n, p=f, observed=falsealarms)

        # Hyperpriors on mRatio
        mu_logMratio = Normal(
            "mu_logMratio", mu=0, tau=1, shape=(1), testval=np.random.rand() * 0.1
        )
        sigma_delta = HalfNormal("sigma_delta", tau=1, shape=(1))

        delta_tilde = Normal("delta_tilde", mu=0, sigma=1, shape=(nSubj, 1))
        delta = Deterministic("delta", sigma_delta * delta_tilde)

        epsilon_logMratio = Beta("epsilon_logMratio", 1, 1, shape=(1))
        logMratio = Deterministic("logMratio", mu_logMratio + epsilon_logMratio * delta)
        mRatio = Deterministic("mRatio", math.exp(logMratio))

        # Type 2 priors
        meta_d = Deterministic("meta_d", mRatio * d1)

        # Specify ordered prior on criteria
        # bounded above and below by Type 1 c1
        cS1_hn = HalfNormal(
            "cS1_hn",
            sigma=sigma_c2,
            shape=(nSubj, nRatings - 1),
            testval=np.linspace(1.5, 0.5, nRatings - 1)
            .reshape(1, nRatings - 1)
            .repeat(nSubj, axis=0),
        )
        cS1 = Deterministic(
            "cS1", -mu_c2 - cS1_hn + (tt.tile(c1, (1, nRatings - 1)) - data["Tol"])
        )

        cS2_hn = HalfNormal(
            "cS2_hn",
            sigma=sigma_c2,
            shape=(nSubj, nRatings - 1),
            testval=np.linspace(0.5, 1.5, nRatings - 1)
            .reshape(1, nRatings - 1)
            .repeat(nSubj, axis=0),
        )
        cS2 = Deterministic(
            "cS2", mu_c2 + cS2_hn + (tt.tile(c1, (1, nRatings - 1)) + data["Tol"])
        )

        # Means of SDT distributions
        S2mu = meta_d / 2
        S1mu = -meta_d / 2

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
                        cumulative_normal(cS1[:, 0].reshape((nSubj, 1)) - S1mu)
                        / C_area_rS1,
                        nC_rS1[:, 1:] - nC_rS1[:, :-1],
                        (
                            (
                                cumulative_normal(c1 - S1mu)
                                - cumulative_normal(
                                    cS1[:, nRatings - 2].reshape((nSubj, 1)) - S1mu
                                )
                            )
                            / C_area_rS1
                        ),
                    ]
                ),
                axis=1,
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
                            - (
                                1
                                - cumulative_normal(
                                    cS2[:, 0].reshape((nSubj, 1)) - S1mu
                                )
                            )
                        )
                        / I_area_rS2,
                        nI_rS2[:, :-1]
                        - (1 - cumulative_normal(cS2[:, 1:] - S1mu)) / I_area_rS2,
                        (
                            1
                            - cumulative_normal(
                                cS2[:, nRatings - 2].reshape((nSubj, 1)) - S1mu
                            )
                        )
                        / I_area_rS2,
                    ]
                ),
                axis=1,
            ),
        )

        # Get nI_rS1 probs
        nI_rS1 = (-cumulative_normal(cS1 - S2mu)) / I_area_rS1
        nI_rS1 = Deterministic(
            "nI_rS1",
            math.concatenate(
                (
                    [
                        cumulative_normal(cS1[:, 0].reshape((nSubj, 1)) - S2mu)
                        / I_area_rS1,
                        nI_rS1[:, :-1]
                        + (cumulative_normal(cS1[:, 1:] - S2mu)) / I_area_rS1,
                        (
                            cumulative_normal(c1 - S2mu)
                            - cumulative_normal(
                                cS1[:, nRatings - 2].reshape((nSubj, 1)) - S2mu
                            )
                        )
                        / I_area_rS1,
                    ]
                ),
                axis=1,
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
                            - (
                                1
                                - cumulative_normal(
                                    cS2[:, 0].reshape((nSubj, 1)) - S2mu
                                )
                            )
                        )
                        / C_area_rS2,
                        nC_rS2[:, :-1]
                        - ((1 - cumulative_normal(cS2[:, 1:] - S2mu)) / C_area_rS2),
                        (
                            1
                            - cumulative_normal(
                                cS2[:, nRatings - 2].reshape((nSubj, 1)) - S2mu
                            )
                        )
                        / C_area_rS2,
                    ]
                ),
                axis=1,
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
            nC_rS1,
            shape=(nSubj, nRatings),
            observed=counts[:, :nRatings],
        )
        Multinomial(
            "FA_counts",
            FA,
            nI_rS2,
            shape=(nSubj, nRatings),
            observed=counts[:, nRatings : nRatings * 2],
        )
        Multinomial(
            "M_counts",
            m,
            nI_rS1,
            shape=(nSubj, nRatings),
            observed=counts[:, nRatings * 2 : nRatings * 3],
        )
        Multinomial(
            "H_counts",
            H,
            nC_rS2,
            shape=(nSubj, nRatings),
            observed=counts[:, nRatings * 3 : nRatings * 4],
        )

        if sample_model is True:

            trace = sample(return_inferencedata=True, **kwargs)

            return model, trace

        else:
            return model
