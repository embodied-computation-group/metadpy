# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import theano as tt
from metadPy.utils import cumulative_normal
from pymc3 import Model, Normal, Binomial, Multinomial, Bound, Deterministic, \
    math, sample


def indiv(data):
    """Compute hierachical meta-d' at the subject level.

    Parameters
    data : dict
        The preprocessed response data.

    Returns
    -------
    model : dict

    Notes
    -----
    
    """
    with Model():

        # Type 1 priors
        c1 = Normal('c1', mu=0.0, tau=2)
        d1 = Normal('d1', mu=0.0, tau=0.5)

        # TYPE 1 SDT BINOMIAL MODEL
        h = cumulative_normal(d1/2-c1)
        f = cumulative_normal(-d1/2-c1)
        H = Binomial('H', data['S'], h, observed=data['H'])
        FA = Binomial('FA', data['N'], f, observed=data['FA'])

        # Type 2 priors
        meta_d = Normal('metad', mu=d1, tau=0.5)

        # Specify ordered prior on criteria (bounded above and below Type 1 c1)
        cS1_raw = Bound(Normal, upper=c1-data['Tol'])('cS1_raw', mu=0.0,
                                                      tau=2.0,
                                                      shape=data['nratings'])
        cS2_raw = Bound(Normal, lower=c1+data['Tol'])('cS2_raw', mu=0.0,
                                                      tau=2.0,
                                                      shape=data['nratings'])

        cS1 = Deterministic('cS1', tt.sort(cS1_raw))
        cS2 = Deterministic('cS2', tt.sort(cS2_raw))

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
        nC_rS1 = tt.set_subtensor(nC_rS1[1:-1], nC_rS1[1:-1] - nC_rS1[:-2])
        nC_rS1 = tt.set_subtensor(
            nC_rS1[0], cumulative_normal(cS1[0] - S1mu)/C_area_rS1)
        nC_rS1 = tt.set_subtensor(
            nC_rS1[-1],
            (cumulative_normal(c1 - S1mu) -
                cumulative_normal(cS1[(data['nratings']-2)] - S1mu))/C_area_rS1)

        # Get nI_rS2 probs
        nI_rS2 = (1-cumulative_normal(cS2 - S1mu))/I_area_rS2
        nI_rS2 = tt.set_subtensor(
            nI_rS2[1:-1], nI_rS2[:-2] - (
                1-cumulative_normal(cS2[1:-1] - S1mu))/I_area_rS2)
        nI_rS2 = tt.set_subtensor(nI_rS2[0], (
            (1-cumulative_normal(c1 - S1mu))-(
                1-cumulative_normal(cS2[0] - S1mu)))/I_area_rS2)
        nI_rS2 = tt.set_subtensor(nI_rS2[-1], (
            1-cumulative_normal(cS2[data['nratings']-2] - S1mu))/I_area_rS2)

        # Get nI_rS1 probs
        nI_rS1 = (-cumulative_normal(cS1 - S2mu))/I_area_rS1
        nI_rS1 = tt.set_subtensor(
            nI_rS1[1:-1],
            nI_rS1[:-2] + (cumulative_normal(cS1[1:-1] - S2mu))/I_area_rS1)
        nI_rS1 = tt.set_subtensor(
            nI_rS1[0], cumulative_normal(cS1[0] - S2mu)/I_area_rS1)
        nI_rS1 = tt.set_subtensor(
            nI_rS1[-1],
            (cumulative_normal(c1 - S2mu) -
                cumulative_normal(cS1[(data['nratings']-2)] - S2mu))/I_area_rS1)

        # Get nC_rS2 probs
        nC_rS2 = (1-cumulative_normal(cS2 - S2mu))/C_area_rS2
        nC_rS2 = tt.set_subtensor(
            nC_rS2[1:-1], nC_rS2[:-2] - (
                (1-cumulative_normal(cS2[1:-1] - S2mu))/C_area_rS2))
        nC_rS2 = tt.set_subtensor(
            nC_rS2[0],
            ((1-cumulative_normal(c1 - S2mu)) -
             (1-cumulative_normal(cS2[0] - S2mu)))/C_area_rS2)
        nC_rS2 = tt.set_subtensor(
            nC_rS2[-1],
            (1-cumulative_normal(cS2[data['nratings']-2] - S2mu))/C_area_rS2)

        # Avoid underflow of probabilities
        nC_rS1 = math.switch(nC_rS1 < data['Tol'], data['Tol'], nC_rS1)
        nI_rS2 = math.switch(nI_rS2 < data['Tol'], data['Tol'], nI_rS2)
        nI_rS1 = math.switch(nI_rS1 < data['Tol'], data['Tol'], nI_rS1)
        nC_rS2 = math.switch(nC_rS2 < data['Tol'], data['Tol'], nC_rS2)

        # TYPE 2 SDT MODEL (META-D)
        # Multinomial likelihood for response counts ordered as c(nR_S1,nR_S2)
        Multinomial(
            'CR_counts', data['CR'],
            nC_rS1,
            shape=data['nratings'],
            observed=data['counts'][:data['nratings']])
        Multinomial(
            'FA_counts', FA,
            nI_rS2,
            shape=data['nratings']*2-data['nratings'],
            observed=data['counts'][data['nratings']:data['nratings']*2])
        Multinomial(
            'M_counts', data['M'],
            nI_rS1,
            shape=data['nratings']*3-(data['nratings']*2),
            observed=data['counts'][data['nratings']*2:data['nratings']*3])
        Multinomial(
            'H_counts', H,
            nC_rS2,
            shape=data['nratings']*4-(data['nratings']*3),
            observed=data['counts'][data['nratings']*3:data['nratings']*4])

        trace = sample(1000, chains=3, progressbar=True,
                       trace=[meta_d, cS1, cS2])

    return {'trace': trace}
