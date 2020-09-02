# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import theano.tensor as tt
from metadPy.sdt import dprime, criterion
import numpy as np
from pymc3 import Model, Normal, Binomial, Multinomial, Bound, Deterministic, \
    math, sample


def cumulative_normal(x):
    """Cummulative normal distribution"""
    return 0.5 + 0.5 * math.erf(x/math.sqrt(2))


def preprocess(nR_S1, nR_S2):
    """Extract rates and task parameters.

    Parameters
    ----------
    nR_S1, nR_S2 : 1d array-like or list
        Total number of responses in each response category, conditional on
        presentation of S1 and S2. e.g. if `nR_S1 = [100 50 20 10 5 1]`, then
        when stimulus S1 was presented, the subject had the following response
        counts:
            * responded S1, rating=3 : 100 times
            * responded S1, rating=2 : 50 times
            * responded S1, rating=1 : 20 times
            * responded S2, rating=1 : 10 times
            * responded S2, rating=2 : 5 times
            * responded S2, rating=3 : 1 time

    Return
    ------
    data : dict
        Dictionnary of rates and task parameters.

    See also
    --------
    hmetad_individual
    """
    Tol = 1e-05
    nratings = int(len(nR_S1)/2)

    # Adjust to ensure non-zero counts for type 1 d' point estimate
    adj_f = 1/((nratings)*2)

    nR_S1_adj = nR_S1 + adj_f
    nR_S2_adj = nR_S2 + adj_f

    ratingHR, ratingFAR = [], []
    for c in range(1, int(nratings*2)):
        ratingHR.append(sum(nR_S2_adj[c:]) / sum(nR_S2_adj))
        ratingFAR.append(sum(nR_S1_adj[c:]) / sum(nR_S1_adj))

    d1 = dprime(ratingHR[nratings-1], ratingFAR[nratings-1])
    c1 = criterion(ratingHR[nratings-1], ratingFAR[nratings-1])
    counts = np.hstack([nR_S1, nR_S2])

    # Type 1 counts
    N = sum(counts[:(nratings*2)])
    S = sum(counts[(nratings*2):(nratings*4)])
    H = sum(counts[(nratings*3):(nratings*4)])
    M = sum(counts[(nratings*2):(nratings*3)])
    FA = sum(counts[(nratings):(nratings*2)])
    CR = sum(counts[:(nratings)])

    # Data preparation for model
    data = {'d1': d1, 'c1': c1, 'counts': counts, 'nratings': nratings,
            'Tol': Tol, 'FA': FA, 'CR': CR, 'M': M, 'H': H, 'N': N, 'S': S}

    return data


def hmetad_individual(data, chains=3, tune=1000, draws=10000):
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
        The number of samples to draw. Defaults to `10000`.

    Returns
    -------
    traces : dict
        Dictionnary of the results and logs:
            * `'trace'`: the MCMC traces

    References
    ----------
    .. [#] Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation of
    metacognitive efficiency from confidence ratings, Neuroscience of
    Consciousness, 3(1) nix007, https://doi.org/10.1093/nc/nix007
    """
    nRatings = data['nratings']
    with Model():

        # Type 1 priors
        c1 = Normal('c1', mu=0.0, tau=2, shape=1)
        d1 = Normal('d1', mu=0.0, tau=0.5, shape=1)

        # TYPE 1 SDT BINOMIAL MODEL
        h = cumulative_normal(d1/2-c1)
        f = cumulative_normal(-d1/2-c1)
        H = Binomial('H', data['S'], h, observed=data['H'])
        FA = Binomial('FA', data['N'], f, observed=data['FA'])

        # Type 2 priors
        meta_d = Normal('metad', mu=d1, tau=2, shape=1)

        # Specify ordered prior on criteria
        # bounded above and below by Type 1 c1
        cS1 = Deterministic(
            'cS1', tt.sort(Bound(Normal, upper=c1-data['Tol'])(
                'cS1_raw', mu=0.0, tau=2, shape=nRatings-1)))
        cS2 = Deterministic(
            'cS2', tt.sort(Bound(Normal, lower=c1+data['Tol'])(
                'cS2_raw', mu=0.0, tau=2, shape=nRatings-1)))

        # Means of SDT distributions
        S2mu = meta_d/2
        S1mu = -meta_d/2

        # Calculate normalisation constants
        C_area_rS1 = cumulative_normal(c1 - S1mu)
        I_area_rS1 = cumulative_normal(c1 - S2mu)
        C_area_rS2 = 1-cumulative_normal(c1 - S2mu)
        I_area_rS2 = 1-cumulative_normal(c1 - S1mu)

        # Get nC_rS1 probs
        nC_rS1 = cumulative_normal(cS1 - S1mu)/C_area_rS1
        nC_rS1 = Deterministic(
            'nC_rS1',
            math.concatenate(
                ([cumulative_normal(cS1[0] - S1mu)/C_area_rS1,
                  nC_rS1[1:] - nC_rS1[:-1],
                  ((cumulative_normal(c1 - S1mu) -
                   cumulative_normal(cS1[(nRatings-2)] - S1mu))/C_area_rS1)]),
                axis=0))

        # Get nI_rS2 probs
        nI_rS2 = (1-cumulative_normal(cS2 - S1mu))/I_area_rS2
        nI_rS2 = Deterministic(
            'nI_rS2',
            math.concatenate(
                ([((1-cumulative_normal(c1 - S1mu)) -
                 (1-cumulative_normal(cS2[0] - S1mu)))/I_area_rS2,
                  nI_rS2[:-1] -
                  (1-cumulative_normal(cS2[1:] - S1mu))/I_area_rS2,
                  (1-cumulative_normal(cS2[nRatings-2] - S1mu))/I_area_rS2]),
                axis=0))

        # Get nI_rS1 probs
        nI_rS1 = (-cumulative_normal(cS1 - S2mu))/I_area_rS1
        nI_rS1 = Deterministic(
            'nI_rS1', math.concatenate(
                ([cumulative_normal(cS1[0] - S2mu)/I_area_rS1,
                  nI_rS1[:-1] + (cumulative_normal(cS1[1:] - S2mu))/I_area_rS1,
                  (cumulative_normal(c1 - S2mu) -
                   cumulative_normal(cS1[(nRatings-2)] - S2mu))/I_area_rS1]),
                axis=0))

        # Get nC_rS2 probs
        nC_rS2 = (1-cumulative_normal(cS2 - S2mu))/C_area_rS2
        nC_rS2 = Deterministic(
            'nC_rS2',
            math.concatenate(
                ([((1-cumulative_normal(c1 - S2mu)) -
                 (1-cumulative_normal(cS2[0] - S2mu)))/C_area_rS2,
                  nC_rS2[:-1] -
                  ((1-cumulative_normal(cS2[1:] - S2mu))/C_area_rS2),
                  (1-cumulative_normal(cS2[nRatings-2] - S2mu))/C_area_rS2]),
                axis=0))

        # Avoid underflow of probabilities
        nC_rS1 = math.switch(nC_rS1 < data['Tol'], data['Tol'], nC_rS1)
        nI_rS2 = math.switch(nI_rS2 < data['Tol'], data['Tol'], nI_rS2)
        nI_rS1 = math.switch(nI_rS1 < data['Tol'], data['Tol'], nI_rS1)
        nC_rS2 = math.switch(nC_rS2 < data['Tol'], data['Tol'], nC_rS2)

        # TYPE 2 SDT MODEL (META-D)
        # Multinomial likelihood for response counts ordered as c(nR_S1,nR_S2)
        Multinomial('CR_counts', data['CR'], nC_rS1,
                    shape=nRatings, observed=data['counts'][:nRatings])
        Multinomial('FA_counts', FA, nI_rS2, shape=nRatings,
                    observed=data['counts'][nRatings:nRatings*2])
        Multinomial('M_counts', data['M'], nI_rS1, shape=nRatings,
                    observed=data['counts'][nRatings*2:nRatings*3])
        Multinomial('H_counts', H, nC_rS2, shape=nRatings,
                    observed=data['counts'][nRatings*3:nRatings*4])

        trace = sample(draws, chains=chains, progressbar=True,
                       trace=[meta_d, cS1, cS2], tune=tune)

    return trace
