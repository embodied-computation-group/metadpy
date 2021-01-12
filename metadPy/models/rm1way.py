# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import theano.tensor as tt
from pymc3 import (
    Model,
    Normal,
    HalfNormal,
    Exponential,
    Multinomial,
    Deterministic,
    math,
    sample,
)


def cumulative_normal(x):
    """Cummulative normal distribution"""
    return 0.5 + 0.5 * math.erf(x / math.sqrt(2))


def hmetad_rm1way(data: dict, sample_model: bool = True, **kwargs: int):
    """Compute hierachical meta-d' at the subject level.

    This is an internal function. The repeated measures model must be
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
    nCond = data["nCond"]
    nRatings = data["nRatings"]
    hits = data["hits"].reshape(nSubj, 2)
    falsealarms = data["falsealarms"].reshape(nSubj, 2)
    counts = data["counts"]
    Tol = data["Tol"]
    cr = data["cr"].reshape(nSubj, 2)
    m = data["m"].reshape(nSubj, 2)
    c1 = data["c1"].reshape(nSubj, 2, 1)
    d1 = data["d1"].reshape(nSubj, 2, 1)

    with Model() as model:

        #############
        # Hyperpriors
        #############
        mu_c2 = Normal(
            "mu_c2", tau=0.01, shape=(1, 1, 1), testval=np.random.rand() * 0.1
        )
        sigma_c2 = HalfNormal(
            "sigma_c2", tau=0.01, shape=(1, 1, 1), testval=np.random.rand() * 0.1
        )

        mu_D = Normal("mu_D", tau=0.001, shape=(1), testval=np.random.rand() * 0.1)
        sigma_D = HalfNormal(
            "sigma_D", tau=0.1, shape=(1), testval=np.random.rand() * 0.1
        )

        mu_Cond1 = Normal(
            "mu_Cond1", mu=0, tau=0.001, shape=(1), testval=np.random.rand() * 0.1
        )
        sigma_Cond1 = HalfNormal(
            "sigma_Cond1", tau=0.1, shape=(1), testval=np.random.rand() * 0.1
        )

        #############################
        # Hyperpriors - Subject level
        #############################
        dbase_tilde = Normal(
            "dbase_tilde",
            mu=0,
            sigma=1,
            shape=(nSubj, 1, 1),
        )
        dbase = Deterministic("dbase", mu_D + sigma_D * dbase_tilde)

        Bd_Cond1 = Normal(
            "Bd_Cond1",
            mu=mu_Cond1,
            sigma=sigma_Cond1,
            shape=(nSubj, 1, 1),
            testval=(np.random.rand(nSubj) * 0.1).reshape(nSubj, 1, 1),
        )

        tau_logMratio = Exponential("tau_logMratio", 0.1, shape=(nSubj, 1, 1))

        ###############################
        # Hypterprior - Condition level
        ###############################
        mu_regression = [dbase + (Bd_Cond1 * c) for c in range(nCond)]

        log_mRatio_tilde = Normal(
            "log_mRatio_tilde", mu=0, tau=tau_logMratio, shape=(nSubj, 1, 1)
        )
        log_mRatio = Deterministic(
            "log_mRatio",
            tt.stack(mu_regression, axis=1)[:, :, :, 0]
            + tt.tile(log_mRatio_tilde, (1, 2, 1)),
        )

        mRatio = Deterministic("mRatio", tt.exp(log_mRatio))

        # Means of SDT distributions
        metad = Deterministic("metad", mRatio * d1)
        S2mu = Deterministic("S2mu", metad / 2)
        S1mu = Deterministic("S1mu", -metad / 2)

        # TYPE 2 SDT MODEL (META-D)
        # Multinomial likelihood for response counts
        # Specify ordered prior on criteria
        # bounded above and below by Type 1 c
        cS1_hn = HalfNormal(
            "cS1_hn",
            sigma=sigma_c2,
            shape=(nSubj, nCond, nRatings - 1),
            testval=np.linspace(1.5, 0.5, nRatings - 1)
            .reshape(1, 1, nRatings - 1)
            .repeat(nSubj, axis=0)
            .repeat(nCond, axis=1),
        )
        cS1 = Deterministic("cS1", -mu_c2 - cS1_hn + (c1 - data["Tol"]))

        cS2_hn = HalfNormal(
            "cS2_hn",
            sigma=sigma_c2,
            shape=(nSubj, nCond, nRatings - 1),
            testval=np.linspace(0.5, 1.5, nRatings - 1)
            .reshape(1, 1, nRatings - 1)
            .repeat(nSubj, axis=0)
            .repeat(nCond, axis=1),
        )
        cS2 = Deterministic("cS2", mu_c2 + cS2_hn + (c1 + data["Tol"]))

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
                        cumulative_normal(cS1[:, :, 0].reshape((nSubj, 2, 1)) - S1mu)
                        / C_area_rS1,
                        nC_rS1[:, :, 1:] - nC_rS1[:, :, :-1],
                        (
                            (
                                cumulative_normal(c1 - S1mu)
                                - cumulative_normal(
                                    cS1[:, :, (nRatings - 2)].reshape((nSubj, 2, 1))
                                    - S1mu
                                )
                            )
                            / C_area_rS1
                        ),
                    ]
                ),
                axis=2,
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
                                    cS2[:, :, 0].reshape((nSubj, nCond, 1)) - S1mu
                                )
                            )
                        )
                        / I_area_rS2,
                        nI_rS2[:, :, :-1]
                        - (1 - cumulative_normal(cS2[:, :, 1:] - S1mu)) / I_area_rS2,
                        (
                            1
                            - cumulative_normal(
                                cS2[:, :, nRatings - 2].reshape((nSubj, nCond, 1))
                                - S1mu
                            )
                        )
                        / I_area_rS2,
                    ]
                ),
                axis=2,
            ),
        )

        # Get nI_rS1 probs
        nI_rS1 = (-cumulative_normal(cS1 - S2mu)) / I_area_rS1
        nI_rS1 = Deterministic(
            "nI_rS1",
            math.concatenate(
                (
                    [
                        cumulative_normal(
                            cS1[:, :, 0].reshape((nSubj, nCond, 1)) - S2mu
                        )
                        / I_area_rS1,
                        nI_rS1[:, :, :-1]
                        + (cumulative_normal(cS1[:, :, 1:] - S2mu)) / I_area_rS1,
                        (
                            cumulative_normal(c1 - S2mu)
                            - cumulative_normal(
                                cS1[:, :, nRatings - 2].reshape((nSubj, nCond, 1))
                                - S2mu
                            )
                        )
                        / I_area_rS1,
                    ]
                ),
                axis=2,
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
                                    cS2[:, :, 0].reshape((nSubj, nCond, 1)) - S2mu
                                )
                            )
                        )
                        / C_area_rS2,
                        nC_rS2[:, :, :-1]
                        - ((1 - cumulative_normal(cS2[:, :, 1:] - S2mu)) / C_area_rS2),
                        (
                            1
                            - cumulative_normal(
                                cS2[:, :, nRatings - 2].reshape((nSubj, nCond, 1))
                                - S2mu
                            )
                        )
                        / C_area_rS2,
                    ]
                ),
                axis=2,
            ),
        )

        # Avoid underflow of probabilities
        nC_rS1 = math.switch(nC_rS1 < Tol, Tol, nC_rS1)
        nI_rS2 = math.switch(nI_rS2 < Tol, Tol, nI_rS2)
        nI_rS1 = math.switch(nI_rS1 < Tol, Tol, nI_rS1)
        nC_rS2 = math.switch(nC_rS2 < Tol, Tol, nC_rS2)

        for c in range(nCond):
            Multinomial(
                f"CR_counts_{c}",
                n=cr[:, c],
                p=nC_rS1[:, c, :],
                observed=counts[:, c, :nRatings],
                shape=(nSubj, nRatings),
            )
            Multinomial(
                f"H_counts_{c}",
                n=hits[:, c],
                p=nC_rS2[:, c, :],
                observed=counts[:, c, nRatings * 3 : nRatings * 4],
                shape=(nSubj, nRatings),
            )
            Multinomial(
                f"FA_counts_{c}",
                n=falsealarms[:, c],
                p=nI_rS2[:, c, :],
                observed=counts[:, c, nRatings : nRatings * 2],
                shape=(nSubj, nRatings),
            )
            Multinomial(
                f"M_counts_{c}",
                n=m[:, c],
                p=nI_rS1[:, c, :],
                observed=counts[:, c, nRatings * 2 : nRatings * 3],
                shape=(nSubj, nRatings),
            )

        if sample_model is True:

            trace = sample(return_inferencedata=True, **kwargs)

            return model, trace

        else:
            return model
