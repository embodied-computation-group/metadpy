# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import erf


def phi(x):
    """Cumulative normal distribution"""
    return 0.5 + 0.5 * erf(x / jnp.sqrt(2))


def hmetad_groupLevel(data: dict):
    """Compute hierachical meta-d' at the subject level.

    This is an internal function. The group level model must be
    called using :py:func:`metadPy.hierarchical.hmetad`.

    Parameters
    ----------
    data : dict
        Response data.

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

    # hyperpriors on d, c and c2
    mu_c1 = numpyro.sample("mu_c1", dist.Normal(0.0, 1 / jnp.sqrt(0.01)))
    mu_c2 = numpyro.sample("mu_c2", dist.Normal(0.0, 1 / jnp.sqrt(0.01)))
    mu_d1 = numpyro.sample("mu_d1", dist.Normal(0.0, 1 / jnp.sqrt(0.01)))

    sigma_c1 = numpyro.sample("sigma_c1", dist.HalfNormal(1 / jnp.sqrt(0.01)))
    sigma_c2 = numpyro.sample("sigma_c2", dist.HalfNormal(1 / jnp.sqrt(0.01)))
    sigma_d1 = numpyro.sample("sigma_d1", dist.HalfNormal(1 / jnp.sqrt(0.01)))

    # Type 1 priors
    with numpyro.plate("plate_i", nSubj):
        c1_tilde = numpyro.sample("c1_tilde", dist.Normal(0.0, 1.0))
        c1 = numpyro.deterministic("c1", mu_c1 + (sigma_c1 * c1_tilde))

        d1_tilde = numpyro.sample("d1_tilde", dist.Normal(0.0, 1.0))
        d1 = numpyro.deterministic("d1", mu_d1 + (sigma_d1 * d1_tilde))

    # TYPE 1 SDT BINOMIAL MODEL
    h = phi((d1 / 2) - c1)
    f = phi((-d1 / 2) - c1)
    H = numpyro.sample("H", dist.Binomial(total_count=s, probs=h), obs=hits)
    FA = numpyro.sample("FA", dist.Binomial(total_count=n, probs=f), obs=falsealarms)

    c1 = jnp.expand_dims(c1, axis=0)

    # Hyperpriors on m_ratio
    mu_log_m_ratio = numpyro.sample("mu_log_m_ratio", dist.Normal(0.0, 1.0))
    mu_m_ratio = numpyro.deterministic("mu_m_ratio", jnp.exp(mu_log_m_ratio))
    mu_meta_d = numpyro.deterministic("mu_meta_d", mu_m_ratio * mu_d1)
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(1.0))

    delta_tilde = numpyro.sample(
        "delta_tilde", dist.Normal(0.0, 1.0), sample_shape=(nSubj,)
    )
    delta = numpyro.deterministic("delta", sigma_delta * delta_tilde)

    epsilon_log_m_ratio = numpyro.sample("epsilon_log_m_ratio", dist.Beta(1, 1))
    log_m_ratio = numpyro.deterministic(
        "log_m_ratio", mu_log_m_ratio + (epsilon_log_m_ratio * delta)
    )
    m_ratio = numpyro.deterministic("m_ratio", jnp.exp(log_m_ratio))

    # Type 2 priors
    meta_d = numpyro.deterministic("meta_d", m_ratio * d1)

    # Specify ordered prior on criteria
    # bounded above and below by Type 1 c1
    with numpyro.plate("plate_j", nSubj):
        with numpyro.plate("plate_l", nRatings - 1):
            cS1_hn = numpyro.sample("cS1_hn", dist.Normal(0.0, 1.0))
            cS2_hn = numpyro.sample("cS2_hn", dist.Normal(0.0, 1.0))

    cS1 = numpyro.deterministic(
        "cS1",
        jnp.clip(
            jnp.sort(-mu_c2 + (cS1_hn * sigma_c2), axis=0),
            a_max=c1.repeat(nRatings - 1, axis=0),
        ),
    )
    cS2 = numpyro.deterministic(
        "cS2",
        jnp.clip(
            jnp.sort(mu_c2 + (cS2_hn * sigma_c2), axis=0),
            a_min=c1.repeat(nRatings - 1, axis=0),
        ),
    )

    # Means of SDT distributions
    S2mu = jnp.expand_dims(meta_d / 2, axis=0)
    S1mu = jnp.expand_dims(-meta_d / 2, axis=0)

    # Calculate normalisation constants
    C_area_rS1 = phi(c1 - S1mu)
    I_area_rS1 = phi(c1 - S2mu)
    C_area_rS2 = 1 - phi(c1 - S2mu)
    I_area_rS2 = 1 - phi(c1 - S1mu)

    # Get nC_rS1 probs
    nC_rS1 = phi(cS1 - S1mu) / C_area_rS1
    part1 = phi(cS1[[0], :] - S1mu) / C_area_rS1
    part2 = nC_rS1[1:, :] - nC_rS1[:-1, :]
    part3 = (phi(c1 - S1mu) - phi(cS1[[nRatings - 2], :] - S1mu)) / C_area_rS1
    nC_rS1 = jnp.concatenate((part1, part2, part3), axis=0)

    # Get nI_rS2 probs
    nI_rS2 = (1 - phi(cS2 - S1mu)) / I_area_rS2
    part1 = ((1 - phi(c1 - S1mu)) - (1 - phi(cS2[[0], :] - S1mu))) / I_area_rS2
    part2 = nI_rS2[:-1, :] - (1 - phi(cS2[1:, :] - S1mu)) / I_area_rS2
    part3 = (1 - phi(cS2[[nRatings - 2], :] - S1mu)) / I_area_rS2
    nI_rS2 = jnp.concatenate((part1, part2, part3), axis=0)

    # Get nI_rS1 probs
    nI_rS1 = (-phi(cS1 - S2mu)) / I_area_rS1
    part1 = phi(cS1[[0], :] - S2mu) / I_area_rS1
    part2 = nI_rS1[:-1, :] + (phi(cS1[1:, :] - S2mu)) / I_area_rS1
    part3 = (phi(c1 - S2mu) - phi(cS1[[nRatings - 2], :] - S2mu)) / I_area_rS1
    nI_rS1 = jnp.concatenate((part1, part2, part3), axis=0)

    # Get nC_rS2 probs
    nC_rS2 = (1 - phi(cS2 - S2mu)) / C_area_rS2
    part1 = ((1 - phi(c1 - S2mu)) - (1 - phi(cS2[[0], :] - S2mu))) / C_area_rS2
    part2 = nC_rS2[:-1, :] - ((1 - phi(cS2[1:, :] - S2mu)) / C_area_rS2)
    part3 = (1 - phi(cS2[[nRatings - 2], :] - S2mu)) / C_area_rS2
    nC_rS2 = jnp.concatenate((part1, part2, part3), axis=0)

    # Avoid underflow of probabilities
    nC_rS1 = jnp.clip(nC_rS1, a_min=Tol)
    nI_rS2 = jnp.clip(nI_rS2, a_min=Tol)
    nI_rS1 = jnp.clip(nI_rS1, a_min=Tol)
    nC_rS2 = jnp.clip(nC_rS2, a_min=Tol)

    # TYPE 2 SDT MODEL (META-D)
    # Multinomial likelihood for response counts ordered as c(nR_S1,nR_S2)
    numpyro.sample(
        "CR_counts",
        dist.Multinomial(
            total_count=cr,
            probs=nC_rS1.swapaxes(0, 1),
        ),
        sample_shape=(nSubj, nRatings),
        obs=counts[:, :nRatings],
    )
    numpyro.sample(
        "FA_counts",
        dist.Multinomial(
            total_count=FA,
            probs=nI_rS2.swapaxes(0, 1),
        ),
        sample_shape=(nSubj, nRatings),
        obs=counts[:, nRatings : nRatings * 2],
    )
    numpyro.sample(
        "M_counts",
        dist.Multinomial(
            total_count=m,
            probs=nI_rS1.swapaxes(0, 1),
        ),
        sample_shape=(nSubj, nRatings),
        obs=counts[:, nRatings * 2 : nRatings * 3],
    )
    numpyro.sample(
        "H_counts",
        dist.Multinomial(
            total_count=H,
            probs=nC_rS2.swapaxes(0, 1),
        ),
        sample_shape=(nSubj, nRatings),
        obs=counts[:, nRatings * 3 : nRatings * 4],
    )
