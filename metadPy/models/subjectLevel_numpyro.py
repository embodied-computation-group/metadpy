# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import erf


def cumulative_normal(x):
    """Cummulative normal distribution"""
    return 0.5 + 0.5 * erf(x / jnp.sqrt(2))


def hmetad_subjectLevel(data, sample_model=True, **kwargs):
    """Hierachical Bayesian modeling of meta-d' (subject level).

    This is an internal function. The subject level model must be
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
    nRatings = data["nratings"]

    # Type 1 priors
    c1 = numpyro.sample("c1", dist.Normal(0.0, 1 / jnp.sqrt(2)))
    d1 = numpyro.sample("d1", dist.Normal(0.0, 1 / jnp.sqrt(0.5)))

    # TYPE 1 SDT BINOMIAL MODEL
    h = cumulative_normal(d1 / 2 - c1)
    f = cumulative_normal(-d1 / 2 - c1)
    H = numpyro.sample(
        "H", dist.Binomial(probs=h, total_count=data["S"]), obs=data["H"]
    )
    FA = numpyro.sample(
        "FA", dist.Binomial(probs=f, total_count=data["N"]), obs=data["FA"]
    )

    # Type 2 priors
    meta_d = numpyro.sample("metad", dist.Normal(d1, 1 / jnp.sqrt(2)))

    # Specify ordered prior on criteria
    # bounded above and below by Type 1 c1
    cS1_hn = numpyro.sample(
        "cS1_hn",
        dist.HalfNormal(1 / jnp.sqrt(2), dist.constraints.ordered_vector),
        sample_shape=(nRatings - 1,),
    )
    cS1 = numpyro.deterministic("cS1", jnp.sort(-cS1_hn) + (c1 - data["Tol"]))

    cS2_hn = numpyro.sample(
        "cS2_hn",
        dist.HalfNormal(1 / jnp.sqrt(2), dist.constraints.ordered_vector),
        sample_shape=(nRatings - 1,),
    )
    cS2 = numpyro.deterministic("cS2", jnp.sort(cS2_hn) + (c1 - data["Tol"]))

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
    part1 = jnp.array([cumulative_normal(cS1[0] - S1mu) / C_area_rS1])
    part2 = nC_rS1[1:] - nC_rS1[:-1]
    part3 = jnp.array(
        [
            (
                cumulative_normal(c1 - S1mu)
                - cumulative_normal(cS1[(nRatings - 2)] - S1mu)
            )
            / C_area_rS1
        ]
    )
    nC_rS1 = jnp.concatenate((part1, part2, part3))

    # Get nI_rS2 probs
    nI_rS2 = (1 - cumulative_normal(cS2 - S1mu)) / I_area_rS2
    part1 = jnp.array(
        [
            (
                (1 - cumulative_normal(c1 - S1mu))
                - (1 - cumulative_normal(cS2[0] - S1mu))
            )
            / I_area_rS2
        ]
    )
    part2 = nI_rS2[:-1] - (1 - cumulative_normal(cS2[1:] - S1mu)) / I_area_rS2
    part3 = jnp.array([(1 - cumulative_normal(cS2[nRatings - 2] - S1mu)) / I_area_rS2])
    nI_rS2 = jnp.concatenate((part1, part2, part3))

    # Get nI_rS1 probs
    nI_rS1 = (-cumulative_normal(cS1 - S2mu)) / I_area_rS1
    part1 = jnp.array([cumulative_normal(cS1[0] - S2mu) / I_area_rS1])
    part2 = nI_rS1[:-1] + (cumulative_normal(cS1[1:] - S2mu)) / I_area_rS1
    part3 = jnp.array(
        [
            (
                cumulative_normal(c1 - S2mu)
                - cumulative_normal(cS1[(nRatings - 2)] - S2mu)
            )
            / I_area_rS1
        ]
    )
    nI_rS1 = jnp.concatenate((part1, part2, part3))

    # Get nC_rS2 probs
    nC_rS2 = (1 - cumulative_normal(cS2 - S2mu)) / C_area_rS2
    part1 = jnp.array(
        [
            (
                (1 - cumulative_normal(c1 - S2mu))
                - (1 - cumulative_normal(cS2[0] - S2mu))
            )
            / C_area_rS2
        ]
    )
    part2 = nC_rS2[:-1] - ((1 - cumulative_normal(cS2[1:] - S2mu)) / C_area_rS2)
    part3 = jnp.array([(1 - cumulative_normal(cS2[nRatings - 2] - S2mu)) / C_area_rS2])
    nC_rS2 = jnp.concatenate((part1, part2, part3))

    # Avoid underflow of probabilities
    nC_rS1 = jnp.clip(nC_rS1, a_min=data["Tol"])
    nI_rS2 = jnp.clip(nI_rS2, a_min=data["Tol"])
    nI_rS1 = jnp.clip(nI_rS1, a_min=data["Tol"])
    nC_rS2 = jnp.clip(nC_rS2, a_min=data["Tol"])

    # TYPE 2 SDT MODEL (META-D)
    # Multinomial likelihood for response counts ordered as c(nR_S1,nR_S2)
    numpyro.sample(
        "CR_counts",
        dist.Multinomial(
            total_count=data["CR"],
            probs=nC_rS1,
        ),
        sample_shape=(nRatings,),
        obs=data["counts"][:nRatings],
    )
    numpyro.sample(
        "FA_counts",
        dist.Multinomial(
            total_count=FA,
            probs=nI_rS2,
        ),
        sample_shape=(nRatings,),
        obs=data["counts"][nRatings : nRatings * 2],
    )
    numpyro.sample(
        "M_counts",
        dist.Multinomial(
            total_count=data["M"],
            probs=nI_rS1,
        ),
        sample_shape=(nRatings,),
        obs=data["counts"][nRatings * 2 : nRatings * 3],
    )
    numpyro.sample(
        "H_counts",
        dist.Multinomial(
            total_count=H,
            probs=nC_rS2,
        ),
        sample_shape=(nRatings,),
        obs=data["counts"][nRatings * 3 : nRatings * 4],
    )
