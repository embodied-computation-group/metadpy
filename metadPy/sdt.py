# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import Bounds, LinearConstraint, minimize, SR1


def scores(data=None, signal='signal', responses='responses'):
    """Extract hits, misses, false alarms and correct rejection from `signal`
    and `responses`.

    Parameters
    ----------
    data :  :py:class:`pandas.DataFrame` or None
        Dataframe containing one `signal` and one `response` column.
    signal : str, 1d array-like or list
        If a string is provided, should be the name of the column used as
        `signal`. If a list or an array is provided, should contain the boolean
        vectors for `signal`.
    responses : str or 1d array-like
        If a string is provided, should be the name of the column used as
        `responses`. If a list or an array is provided, should contain the
        boolean vector for `responses`.

    Returns
    -------
    hits, misses, fas, crs : floats
        Return the number of hits, misees, false alarms and correct rejections.

    Notes
    -----
    If a :py:class:`pandas.DataFrame` is provided, the function will search for
    a `signal`and a `responses` column by default. Other columns names ca be
    provided.
    """
    # Formatting checks
    if data is None:
        if isinstance(signal, (np.ndarray, np.generic)) and \
           isinstance(responses, (np.ndarray, np.generic)) and \
           (len(signal) == len(responses)) and \
           np.all([s in [0, 1] for s in signal]) and\
           np.all([s in [0, 1] for s in responses]):

            data = pd.DataFrame({'signal': signal, 'responses': responses})
        else:
            raise ValueError(('If no data is provided, `signal` and',
                              ' `responses` should be two boolean vectors',
                              ' with equal lengths.'))

    # Extract hits, misses, false alarm and correct rejection
    hit = sum(data['signal'] & data['responses'])
    misses = sum(data['signal'] & ~data['responses'])
    fa = sum(~data['signal'] & data['responses'])
    cr = sum(~data['signal'] & ~data['responses'])

    return hit, misses, fa, cr


def rates(hits, misses, fas, crs):
    """Hit and false alarm rates.

    Parameters
    ----------
    hits : float
        Hits.
    misses :  float
        Misses.
    fas : float
        False alarms.
    crs : float
        Correct rejections.

    Returns
    -------
    hit_rate, fa_rate : float
        Hit and false alarm rate.

    References
    ----------
    Adapted from: https://lindeloev.net/calculating-d-in-python-and-php/
    """
    # Floors an ceilings are replaced with half inverse hits and fa
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)

    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    if hit_rate == 1:
        hit_rate = 1 - half_hit
    if hit_rate == 0:
        hit_rate = half_hit

    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    if fa_rate == 1:
        fa_rate = 1 - half_fa
    if fa_rate == 0:
        fa_rate = half_fa

    return hit_rate, fa_rate


def dprime(hit_rate, fa_rate):
    """Calculate d'.

    Parameters
    ----------
    hit_rate : float
        Hit rate.
    fa_rate : float
        False alarm rate.

    Returns
    -------
    dprime : float
        The d' value.

    Notes
    -----
    The d’ is a measure of the ability to discriminate a signal from noise.
    """
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)


def criterion(hit_rate, fa_rate):
    """Calculate criterion.
    Parameters
    ----------
    hit_rate : float
        Hit rate.
    fa_rate : float
        False alarm rate.

    Returns
    -------
    dprime : float
        The d' value.
    """
    return - .5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))


def fit_meta_d_logL(parameters, inputObj):
    """Returns negative log-likelihood of parameters given experimental data.

    Parameters
    ----------
    parameters : list
        parameters[0] = meta d'
        parameters[1:end] = type-2 criteria locations
    """
    meta_d1 = parameters[0]
    t2c1 = parameters[1:]
    nR_S1, nR_S2, nRatings, d1, t1c1, s, constant_criterion, fncdf, fninv = \
        inputObj

    # define mean and SD of S1 and S2 distributions
    S1mu = -meta_d1/2
    S1sd = 1
    S2mu = meta_d1/2
    S2sd = S1sd/s

    # adjust so that the type 1 criterion is set at 0
    # (this is just to work with optimization toolbox constraints...
    #  to simplify defining the upper and lower bounds of type 2 criteria)
    S1mu = S1mu - eval(constant_criterion)
    S2mu = S2mu - eval(constant_criterion)

    t1c1 = 0

    # set up MLE analysis
    # get type 2 response counts
    # S1 responses
    nC_rS1 = [nR_S1[i] for i in range(nRatings)]
    nI_rS1 = [nR_S2[i] for i in range(nRatings)]
    # S2 responses
    nC_rS2 = [nR_S2[i+nRatings] for i in range(nRatings)]
    nI_rS2 = [nR_S1[i+nRatings] for i in range(nRatings)]

    # get type 2 probabilities
    C_area_rS1 = fncdf(t1c1, S1mu, S1sd)
    I_area_rS1 = fncdf(t1c1, S2mu, S2sd)

    C_area_rS2 = 1-fncdf(t1c1, S2mu, S2sd)
    I_area_rS2 = 1-fncdf(t1c1, S1mu, S1sd)

    t2c1x = [-np.inf]
    t2c1x.extend(t2c1[0:(nRatings-1)])
    t2c1x.append(t1c1)
    t2c1x.extend(t2c1[(nRatings-1):])
    t2c1x.append(np.inf)

    prC_rS1 = [(fncdf(t2c1x[i+1], S1mu, S1sd) -
               fncdf(t2c1x[i], S1mu, S1sd)) / C_area_rS1
               for i in range(nRatings)]
    prI_rS1 = [(fncdf(t2c1x[i+1], S2mu, S2sd) -
               fncdf(t2c1x[i], S2mu, S2sd)) / I_area_rS1
               for i in range(nRatings)]

    prC_rS2 = [((1-fncdf(t2c1x[nRatings+i], S2mu, S2sd)) -
               (1-fncdf(t2c1x[nRatings+i+1], S2mu, S2sd))) / C_area_rS2
               for i in range(nRatings)]
    prI_rS2 = [((1-fncdf(t2c1x[nRatings+i], S1mu, S1sd)) -
               (1-fncdf(t2c1x[nRatings+i+1], S1mu, S1sd))) / I_area_rS2
               for i in range(nRatings)]

    # calculate logL
    logL = np.sum([
            nC_rS1[i]*np.log(prC_rS1[i])
            + nI_rS1[i]*np.log(prI_rS1[i])
            + nC_rS2[i]*np.log(prC_rS2[i])
            + nI_rS2[i]*np.log(prI_rS2[i]) for i in range(nRatings)])

    if np.isinf(logL) or np.isnan(logL):
        logL = -1e+300  # returning -inf may cause optimize.minimize() to fail
    return -logL


def fit_meta_d_MLE(nR_S1, nR_S2, s=1, fncdf=norm.cdf, fninv=norm.ppf):
    """Estimate meta-d' using maximum likelihood estimation (MLE).

    This function is adapted from the transcription of fit_meta_d_MLE.m
    (Maniscalco & Lau, 2012) by Alan Lee
    (http://www.columbia.edu/~bsm2105/type2sdt/) with minor changes.

    Parameters
    ----------
    nR_S1, nR_S2 : list or 1d array-like
        These are vectors containing the total number of responses in
        each response category, conditional on presentation of S1 and S2. If
        nR_S1 = [100 50 20 10 5 1], then when stimulus S1 was presented, the
        subject had the following response counts:
            * responded `'S1'`, rating=`3` : 100 times
            * responded `'S1'`, rating=`2` : 50 times
            * responded `'S1'`, rating=`1` : 20 times
            * responded `'S2'`, rating=`1` : 10 times
            * responded `'S2'`, rating=`2` : 5 times
            * responded `'S2'`, rating=`3` : 1 time

        The ordering of response / rating counts for S2 should be the same as
        it is for S1. e.g. if nR_S2 = [3 7 8 12 27 89], then when stimulus S2
        was presented, the subject had the following response counts:
            * responded `'S1'`, rating=`3` : 3 times
            * responded `'S1'`, rating=`2` : 7 times
            * responded `'S1'`, rating=`1` : 8 times
            * responded `'S2'`, rating=`1` : 12 times
            * responded `'S2'`, rating=`2` : 27 times
            * responded `'S2'`, rating=`3` : 89 times

    Returns
    -------
    fit : dict
        In the following, S1 and S2 represent the distributions of evidence
        generated by stimulus classes S1 and S2:

            * `'da'` : `mean(S2) - mean(S1)`, in
                root-mean-square(sd(S1), sd(S2)) units
            * `'s'` : `sd(S1) / sd(S2)`
            * `'meta_da'` : meta-d' in RMS units
            * `'M_diff'` : `meta_da - da`
            * `'M_ratio'` : `meta_da / da`
            * `'meta_ca'` : type 1 criterion for meta-d' fit, RMS units.
            * `'t2ca_rS1'` : type 2 criteria of "S1" responses for meta-d' fit,
                RMS units.
            * `'t2ca_rS2'` : type 2 criteria of "S2" responses for meta-d' fit,
                RMS units.

            * `'S1units'` : contains same parameters in sd(S1) units. These may
                be of use since the data-fitting is conducted using parameters
                specified in sd(S1) units.

            * `'logL'` : log likelihood of the data fit
            * `'est_HR2_rS1'` : estimated (from meta-d' fit) type 2 hit rates
                for S1 responses.
            * `'obs_HR2_rS1'` : actual type 2 hit rates for S1 responses.
            * `'est_FAR2_rS1'` : estimated type 2 false alarm rates for S1
                responses.
            * `'obs_FAR2_rS1'` : actual type 2 false alarm rates for S1
                responses.

            * `'est_HR2_rS2'` : estimated type 2 hit rates for S2 responses.
            * `'obs_HR2_rS2'` : actual type 2 hit rates for S2 responses.
            * `'est_FAR2_rS2'` : estimated type 2 false alarm rates for S2
                responses.
            * `'obs_FAR2_rS2'` : actual type 2 false alarm rates for S2
                responses.

    Notes
    -----
    Given data from an experiment where an observer discriminates between two
    stimulus alternatives on every trial and provides confidence ratings,
    provides a type 2 SDT analysis of the data.

    N.B. if nR_S1 or nR_S2 contain zeros, this may interfere with estimation of
    meta-d'.

    Some options for dealing with response cell counts containing zeros are:

    (1) Add a small adjustment factor, e.g. adj_f = 1/(length(nR_S1), to each
    input vector:

        adj_f = 1/length(nR_S1)
        nR_S1_adj = nR_S1 + adj_f
        nR_S2_adj = nR_S2 + adj_f

    This is a generalization of the correction for similar estimation issues of
    type 1 d' as recommended in [#]_

    When using this correction method, it is recommended to add the adjustment
    factor to ALL data for all subjects, even for those subjects whose data is
    not in need of such correction, in order to avoid biases in the analysis
    (cf Snodgrass & Corwin, 1988).

    (2) Collapse across rating categories.

    e.g. if your data set has 4 possible confidence ratings such that
    `len(nR_S1)==8`, defining new input vectors:

    >>> nR_S1_new = [sum(nR_S1[:2]), sum(nR_S1[2:4]),
    >>>              sum(nR_S1(5:6)), sum(nR_S1[6:8])]
    >>> nR_S2_new = [sum(nR_S2[:2]), sum(nR_S2[2:4]),
    >>>              sum(nR_S2[4:6]), sum(nR_S2[6:8])]

    might be sufficient to eliminate zeros from the input without using an
    adjustment.

    * s
    this is the ratio of standard deviations for type 1 distributions, i.e.
    `s = sd(S1) / sd(S2)`

    if not specified, s is set to a default value of 1.
    For most purposes, we recommend setting s = 1.
    See http://www.columbia.edu/~bsm2105/type2sdt for further discussion.

    * fncdf
    a function handle for the CDF of the type 1 distribution.
    if not specified, fncdf defaults to @normcdf (i.e. CDF for normal
    distribution)

    * fninv
    a function handle for the inverse CDF of the type 1 distribution.
    if not specified, fninv defaults to @norminv

    If there are N ratings, then there will be N-1 type 2 hit rates and false
    alarm rates.

    Examples
    --------
    >>> nR_S1 = [36, 24, 17, 20, 10, 12, 9, 2]
    >>> nR_S2 = [1, 4, 10, 11, 19, 18, 28, 39]
    >>> fit = fit_meta_d_MLE(nR_S1,nR_S2)

    References
    ---------
    ..[#] Hautus, M. J. (1995). Corrections for extreme proportions and their
    biasing effects on estimated values of d'. Behavior Research Methods,
    Instruments, & Computers, 27, 46-51.
    """
    if (len(nR_S1) % 2) != 0:
        raise ValueError('input arrays must have an even number of elements')
    if len(nR_S1) != len(nR_S2):
        raise ValueError('input arrays must have the same number of elements')
    if any(np.array(nR_S1) == 0) or any(np.array(nR_S2) == 0):
        raise Warning(
            f'Your inputs nR_S1: {nR_S1},  nR_S2: {nR_S2} contain',
            ' zeros! This may interfere with proper estimation of meta-d''.',
            ' See `help fit_meta_d_MLE` for more information.')

    nRatings = int(len(nR_S1) / 2)  # number of ratings in the experiment
    nCriteria = int(2*nRatings - 1)  # number criteria to be fitted

    # parameters
    # meta-d' - 1
    # t2c     - nCriteria-1
    # constrain type 2 criteria values,
    # such that t2c(i) is always <= t2c(i+1)
    # want t2c(i)   <= t2c(i+1)
    # -->  t2c(i+1) >= t2c(i) + 1e-5 (i.e. very small deviation from equality)
    # -->  t2c(i) - t2c(i+1) <= -1e-5
    A, ub, lb = [], [], []
    for ii in range(nCriteria-2):
        tempArow = []
        tempArow.extend(np.zeros(ii+1))
        tempArow.extend([1, -1])
        tempArow.extend(np.zeros((nCriteria-2)-ii-1))
        A.append(tempArow)
        ub.append(-1e-5)
        lb.append(-np.inf)

    # lower bounds on parameters
    LB = []
    LB.append(-10.)                              # meta-d'
    LB.extend(-20*np.ones((nCriteria-1)//2))    # criteria lower than t1c
    LB.extend(np.zeros((nCriteria-1)//2))       # criteria higher than t1c

    # upper bounds on parameters
    UB = []
    UB.append(10.)                           # meta-d'
    UB.extend(np.zeros((nCriteria-1)//2))      # criteria lower than t1c
    UB.extend(20*np.ones((nCriteria-1)//2))    # criteria higher than t1c

    # select constant criterion type
    constant_criterion = 'meta_d1 * (t1c1 / d1)'  # relative criterion

    # set up initial guess at parameter values
    ratingHR = []
    ratingFAR = []
    for c in range(1, int(nRatings*2)):
        ratingHR.append(sum(nR_S2[c:]) / sum(nR_S2))
        ratingFAR.append(sum(nR_S1[c:]) / sum(nR_S1))

    # obtain index in the criteria array to mark Type I and Type II criteria
    t1_index = nRatings-1
    t2_index = list(set(list(range(0, 2*nRatings-1))) - set([t1_index]))

    d1 = (1/s) * fninv(ratingHR[t1_index]) - fninv(ratingFAR[t1_index])
    meta_d1 = d1

    c1 = (-1/(1+s)) * (fninv(ratingHR) + fninv(ratingFAR))
    t1c1 = c1[t1_index]
    t2c1 = c1[t2_index]

    # initial values for the minimization function
    guess = [meta_d1]
    guess.extend(list(t2c1 - eval(constant_criterion)))

    # other inputs for the minimization function
    inputObj = [
        nR_S1, nR_S2, nRatings, d1, t1c1, s, constant_criterion, fncdf, fninv]
    bounds = Bounds(LB, UB)
    linear_constraint = LinearConstraint(A, lb, ub)

    # minimization of negative log-likelihood
    results = minimize(fit_meta_d_logL, guess, args=(inputObj),
                       method='trust-constr', jac='2-point', hess=SR1(),
                       constraints=[linear_constraint],
                       options={'verbose': 1}, bounds=bounds)

    # quickly process some of the output
    meta_d1 = results.x[0]
    t2c1 = results.x[1:] + eval(constant_criterion)
    logL = -results.fun

    # data is fit, now to package it...
    # find observed t2FAR and t2HR

    # I_nR and C_nR are rating trial counts for incorrect and correct trials
    # element i corresponds to # (in)correct w/ rating i
    I_nR_rS2 = nR_S1[nRatings:]
    I_nR_rS1 = list(np.flip(nR_S2[0:nRatings], axis=0))

    C_nR_rS2 = nR_S2[nRatings:]
    C_nR_rS1 = list(np.flip(nR_S1[0:nRatings], axis=0))

    obs_FAR2_rS2 = [
        sum(I_nR_rS2[(i+1):]) / sum(I_nR_rS2) for i in range(nRatings-1)]
    obs_HR2_rS2 = [
        sum(C_nR_rS2[(i+1):]) / sum(C_nR_rS2) for i in range(nRatings-1)]
    obs_FAR2_rS1 = [
        sum(I_nR_rS1[(i+1):]) / sum(I_nR_rS1) for i in range(nRatings-1)]
    obs_HR2_rS1 = [
        sum(C_nR_rS1[(i+1):]) / sum(C_nR_rS1) for i in range(nRatings-1)]

    # find estimated t2FAR and t2HR
    S1mu = -meta_d1/2
    S1sd = 1
    S2mu = meta_d1/2
    S2sd = S1sd/s

    mt1c1 = eval(constant_criterion)

    C_area_rS2 = 1 - fncdf(mt1c1, S2mu, S2sd)
    I_area_rS2 = 1 - fncdf(mt1c1, S1mu, S1sd)

    C_area_rS1 = fncdf(mt1c1, S1mu, S1sd)
    I_area_rS1 = fncdf(mt1c1, S2mu, S2sd)

    est_FAR2_rS2, est_HR2_rS2 = [], []
    est_FAR2_rS1, est_HR2_rS1 = [], []

    for i in range(nRatings-1):

        t2c1_lower = t2c1[(nRatings-1)-(i+1)]
        t2c1_upper = t2c1[(nRatings-1)+i]

        I_FAR_area_rS2 = 1-fncdf(t2c1_upper, S1mu, S1sd)
        C_HR_area_rS2 = 1-fncdf(t2c1_upper, S2mu, S2sd)

        I_FAR_area_rS1 = fncdf(t2c1_lower, S2mu, S2sd)
        C_HR_area_rS1 = fncdf(t2c1_lower, S1mu, S1sd)

        est_FAR2_rS2.append(I_FAR_area_rS2 / I_area_rS2)
        est_HR2_rS2.append(C_HR_area_rS2 / C_area_rS2)

        est_FAR2_rS1.append(I_FAR_area_rS1 / I_area_rS1)
        est_HR2_rS1.append(C_HR_area_rS1 / C_area_rS1)

    # package output
    fit = {}
    fit['da'] = np.sqrt(2/(1+s**2)) * s * d1

    fit['s'] = s

    fit['meta_da'] = np.sqrt(2/(1+s**2)) * s * meta_d1

    fit['M_diff'] = fit['meta_da'] - fit['da']

    fit['M_ratio'] = fit['meta_da'] / fit['da']

    mt1c1 = eval(constant_criterion)
    fit['meta_ca'] = (np.sqrt(2)*s / np.sqrt(1+s**2)) * mt1c1

    t2ca = (np.sqrt(2)*s / np.sqrt(1+s**2)) * np.array(t2c1)
    fit['t2ca_rS1'] = t2ca[0:nRatings-1]
    fit['t2ca_rS2'] = t2ca[(nRatings-1):]

    fit['S1units'] = {}
    fit['S1units']['d1'] = d1
    fit['S1units']['meta_d1'] = meta_d1
    fit['S1units']['s'] = s
    fit['S1units']['meta_c1'] = mt1c1
    fit['S1units']['t2c1_rS1'] = t2c1[0:nRatings-1]
    fit['S1units']['t2c1_rS2'] = t2c1[(nRatings-1):]

    fit['logL'] = logL

    fit['est_HR2_rS1'] = est_HR2_rS1
    fit['obs_HR2_rS1'] = obs_HR2_rS1

    fit['est_FAR2_rS1'] = est_FAR2_rS1
    fit['obs_FAR2_rS1'] = obs_FAR2_rS1

    fit['est_HR2_rS2'] = est_HR2_rS2
    fit['obs_HR2_rS2'] = obs_HR2_rS2

    fit['est_FAR2_rS2'] = est_FAR2_rS2
    fit['obs_FAR2_rS2'] = obs_FAR2_rS2

    return fit


def roc_auc(nR_S1, nR_S2):
    """ Function to calculate the area under the type 2 ROC curve.

    Parameters
    ----------
    nR_S1 : 1d array-like
        Confience ratings (stimuli 1).
    nR_S2 : 1d array-like
        Confidence ratings (stimuli 2).

    Returns
    -------
    rocauc : float
        Area under the type 2 ROC curve.

    Examples
    --------
    """
    nRatings = int(len(nR_S1)/2)

    flip_nR_S1 = np.flip(nR_S1)
    flip_nR_S2 = np.flip(nR_S2)

    S1_H2, S2_H2, S1_FA2, S2_FA2 = [], [], [], []
    for c in range(nRatings):
        S1_H2.append(nR_S1[c] + 0.5)
        S2_H2.append(flip_nR_S2[c] + 0.5)
        S1_FA2.append(flip_nR_S1[c] + 0.5)
        S2_FA2.append(nR_S2[c] + 0.5)

    H2 = S1_H2 + S2_H2
    FA2 = S1_FA2 + S2_FA2

    H2 /= sum(H2)
    FA2 /= sum(FA2)
    cum_H2 = np.hstack((0, np.cumsum(H2)))
    cum_FA2 = np.hstack((0, np.cumsum(FA2)))

    k = []
    for c in range(nRatings):
        k.append((cum_H2[c+1] - cum_FA2[c])**2 - (cum_H2[c] - cum_FA2[c+1])**2)

    return 0.5 + 0.25 * sum(k)
