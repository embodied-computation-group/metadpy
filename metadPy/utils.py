# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
from scipy.stats import norm


def trials2counts(stimID, response, rating, nRatings, padCells=False,
                  padAmount=None):
    '''Response count.

    Given data from an experiment where an observer discriminates between two
    stimulus alternatives on every trial and provides confidence ratings,
    converts trial by trial experimental information for N trials into response
    counts.

    Parameters
    ----------
    stimID : list or 1d array-like
        Stimuli ID.
    response : list or 1d array-like
        Responses.
    rating : list or 1d array-like
        Ratings.
    nRatings : int
        Total of available subjective ratings available for the subject. e.g.
        if subject can rate confidence on a scale of 1-4, then nRatings = 4.
    padCells : boolean
        If `True`, each response count in the output has the value of padAmount
        added to it. Padding cells is desirable if trial counts of 0 interfere
        with model fitting. If False, trial counts are not manipulated and 0s
        may be present in the response count output. Default value for padCells
        is 0.
    padAmount : float
        The value to add to each response count if padCells is set to 1.
        Default value is 1/(2*nRatings)

    Returns
    -------
    nR_S1, nR_S2 : list
        Vectors containing the total number of responses in each response
        category, conditional on presentation of S1 and S2.

    Notes
    -----
    All trials where stimID is not 0 or 1, response is not 0 or 1, or
    rating is not in the range [1, nRatings], are omitted from the response
    count.

    If nR_S1 = [100 50 20 10 5 1], then when stimulus S1 was presented, the
    subject had the following response counts:
        responded S1, rating=3 : 100 times
        responded S1, rating=2 : 50 times
        responded S1, rating=1 : 20 times
        responded S2, rating=1 : 10 times
        responded S2, rating=2 : 5 times
        responded S2, rating=3 : 1 time

    The ordering of response / rating counts for S2 should be the same as it
    is for S1. e.g. if nR_S2 = [3 7 8 12 27 89], then when stimulus S2 was
    presented, the subject had the following response counts:
        responded S1, rating=3 : 3 times
        responded S1, rating=2 : 7 times
        responded S1, rating=1 : 8 times
        responded S2, rating=1 : 12 times
        responded S2, rating=2 : 27 times
        responded S2, rating=3 : 89 times

    Examples
    --------
    >>> stimID = [0, 1, 0, 0, 1, 1, 1, 1]
    >>> response = [0, 1, 1, 1, 0, 0, 1, 1]
    >>> rating = [1, 2, 3, 4, 4, 3, 2, 1]
    >>> nRatings = 4

    >>> nR_S1, nR_S2 = trials2counts(stimID, response, rating, nRatings)
    >>> print(nR_S1, nR_S2)

    Reference
    ---------
    This function was adapted from Alan Lee version of trials2counts.m by
    Maniscalco & Lau (2012) with minor changes.
    '''
    # Check for valid inputs
    if ((len(stimID) == len(response)) and
       (len(stimID) == len(rating))) is False:
        raise ValueError('Input vectors must have the same length')

    tempstim, tempresp, tempratg = [], [], []

    for s, rp, rt in zip(stimID, response, rating):
        if ((s == 0 or s == 1) and
           (rp == 0 or rp == 1) and (rt >= 1 and rt <= nRatings)):
            tempstim.append(s)
            tempresp.append(rp)
            tempratg.append(rt)
    stimID = tempstim
    response = tempresp
    rating = tempratg

    if padAmount is None:
        padAmount = 1/(2*nRatings)

    nR_S1, nR_S2 = [], []

    # S1 responses
    for r in range(nRatings, 0, -1):
        cs1, cs2 = 0, 0
        for s, rp, rt in zip(stimID, response, rating):
            if s == 0 and rp == 0 and rt == r:
                cs1 += 1
            if s == 1 and rp == 0 and rt == r:
                cs2 += 1
        nR_S1.append(cs1)
        nR_S2.append(cs2)

    # S2 responses
    for r in range(1, nRatings+1, 1):
        cs1, cs2 = 0, 0
        for s, rp, rt in zip(stimID, response, rating):
            if s == 0 and rp == 1 and rt == r:
                cs1 += 1
            if s == 1 and rp == 1 and rt == r:
                cs2 += 1
        nR_S1.append(cs1)
        nR_S2.append(cs2)

    # pad response counts to avoid zeros
    if padCells:
        nR_S1 = [n+padAmount for n in nR_S1]
        nR_S2 = [n+padAmount for n in nR_S2]

    return nR_S1, nR_S2


def discreteRatings(ratings, nbins=4):
    """Convert continuous ratings to dscrete bins

    Resample if quantiles are equal at high or low end to ensure proper
    assignment of binned confidence

    Parameters
    ----------
    ratings : list or 1d array-like
        Ratings on a continuous scale.
    nbins : int
        The number of discrete ratings to resample. Defaut set to `4`.

    Returns
    -------
    discreteRatings : 1d array-like
        New rating array only containing integers between 1 and `nbins`.
    out : dict
        Dictionnary containing logs of the discrization process:
            * `'confbins'`: list or 1d array-like - If the ratings were
                reampled, a list containing the new ratings and the new low or
                hg threshold, appened before or after the rating, respectively.
                Else, only returns the ratings.
            * `'rebin'`: boolean - If True, the ratings were resampled due to
                larger numbers of highs or low ratings.
            * `'binCount'` : int - Number of bins

    Examples
    --------
    >>> from metadPy.utils import discreteRatings
    >>> ratings = np.array([
    >>>     96, 98, 95, 90, 32, 58, 77,  6, 78, 78, 62, 60, 38, 12,
    >>>     63, 18, 15, 13, 49, 26,  2, 38, 60, 23, 25, 39, 22, 33,
    >>>     32, 27, 40, 13, 35, 16, 35, 73, 50,  3, 40, 0, 34, 47,
    >>>     52,  0,  0,  0, 25,  1, 16, 37, 59, 20, 25, 23, 45, 22,
    >>>     28, 62, 61, 69, 20, 75, 10, 18, 61, 27, 63, 22, 54, 30,
    >>>     36, 66, 14,  2, 53, 58, 88, 23, 77, 54])
    >>> discreteRatings, out = discreteRatings(ratings)
    (array([4, 4, 4, 4, 2, 3, 4, 1, 4, 4, 4, 4, 3, 1, 4, 1, 1, 1, 3, 2, 1, 3,
        4, 2, 2, 3, 2, 2, 2, 2, 3, 1, 3, 1, 3, 4, 3, 1, 3, 1, 2, 3, 3, 1,
        1, 1, 2, 1, 1, 3, 3, 2, 2, 2, 3, 2, 2, 4, 4, 4, 2, 4, 1, 1, 4, 2,
        4, 2, 3, 2, 3, 4, 1, 1, 3, 3, 4, 2, 4, 3]),
    {'confBins': array([ 0., 20., 35., 60., 98.]), 'rebin': 0, 'binCount': 21})
    """
    out, temp = {}, []
    confBins = np.quantile(ratings, np.linspace(0, 1, nbins+1))
    if (confBins[0] == confBins[1]) & (confBins[nbins-1] == confBins[nbins]):
        raise ValueError('Bad bins!')
    elif confBins[nbins-1] == confBins[nbins]:
        print('Lots of high confidence ratings')
        # Exclude high confidence trials and re-estimate
        hiConf = confBins[-1]
        confBins = np.quantile(ratings[ratings != hiConf],
                               np.linspace(0, 1, nbins))
        for b in range(len(confBins)-1):
            temp.append(
                (ratings >= confBins[b]) & (ratings <= confBins[b+1]))
        temp.append(ratings == hiConf)

        out['confBins'] = [confBins, hiConf]
        out['rebin'] = 1
    elif confBins[0] == confBins[1]:
        print('Lots of low confidence ratings')
        # Exclude low confidence trials and re-estimate
        lowConf = confBins[1]
        temp.append(ratings == lowConf)
        confBins = np.quantile(ratings[ratings != lowConf],
                               np.linspace(0, 1, nbins))
        for b in range(1, len(confBins)):
            temp.append(
                (ratings >= confBins[b-1]) & (ratings <= confBins[b]))
        out['confBins'] = [lowConf, confBins]
        out['rebin'] = 1
    else:
        for b in range(len(confBins)-1):
            temp.append(
                (ratings >= confBins[b]) & (ratings <= confBins[b+1]))
        out['confBins'] = confBins
        out['rebin'] = 0

    discreteRatings = np.zeros(len(ratings), dtype='int')
    for b in range(nbins):
        discreteRatings[temp[b]] = b
    discreteRatings += 1
    out['binCount'] = sum(temp[b])

    return discreteRatings, out


def responseSimulation(d=1, metad=2, c=0, nRatings=4, nTrials=500,
                       as_df=False):
    """ Simulate nR_S1 and nR_S2 response counts.

    Parameters
    ----------
    d : float
        Type 1 dprime.
    metad : float
        Type 2 sensitivity in units of type 1 dprime.
    c : float
        Type 1 criterion.
    nRatings : int
        Number of ratings.
    nTrials : int
        Number of trials to simulate, assumes equal S/N.

    Returns
    -------
    If `as_df is False`:
        nR_S1, nR_S2 : 1d array-like
            nR_S1 and nR_S2 response counts.
    If `as_df is True`:
        df : :py:class:`pandas.DataFrame`
            A DataFrame (nRows==`nTrials`) containing the responses and
            confidence rating for one participant given the provided
            parameters.

    References
    ----------
    Adapted from the Matlab `cpc_metad_sim` function from:
    https://github.com/metacoglab/HMeta-d/blob/master/CPC_metacog_tutorial/cpc_metacog_utils/cpc_metad_sim.m

    See also
    --------
    ratings2df
    """
    # Specify the confidence criterions based on the number of ratings
    c1 = c + np.linspace(-1.5, -0.5, (nRatings - 1))
    c2 = c + np.linspace(0.5, 1.5, (nRatings - 1))

    # Calc type 1 response counts
    H = round((1-norm.cdf(c, d/2))*(nTrials/2))
    FA = round((1-norm.cdf(c, -d/2))*(nTrials/2))
    CR = round(norm.cdf(c, -d/2)*(nTrials/2))
    M = round(norm.cdf(c, d/2)*(nTrials/2))

    # Calc type 2 probabilities
    S1mu = -metad/2
    S2mu = metad/2

    # Normalising constants
    C_area_rS1 = norm.cdf(c, S1mu)
    I_area_rS1 = norm.cdf(c, S2mu)
    C_area_rS2 = 1-norm.cdf(c, S2mu)
    I_area_rS2 = 1-norm.cdf(c, S1mu)

    t2c1x = np.hstack((-np.inf, c1, c, c2, np.inf))

    prC_rS1, prI_rS1, prC_rS2, prI_rS2 = [], [], [], []
    for i in range(nRatings):
        prC_rS1.append((norm.cdf(t2c1x[i+1], S1mu) -
                       norm.cdf(t2c1x[i], S1mu))/C_area_rS1)
        prI_rS1.append((norm.cdf(t2c1x[i+1], S2mu) -
                       norm.cdf(t2c1x[i], S2mu))/I_area_rS1)
        prC_rS2.append(((1-norm.cdf(t2c1x[nRatings+i], S2mu)) -
                       (1-norm.cdf(t2c1x[nRatings+i+1], S2mu)))/C_area_rS2)
        prI_rS2.append(((1-norm.cdf(t2c1x[nRatings+i], S1mu)) -
                       (1-norm.cdf(t2c1x[nRatings+i+1], S1mu)))/I_area_rS2)

    # Ensure vectors sum to 1 to avoid problems with mnrnd
    prC_rS1 = prC_rS1/sum(prC_rS1)
    prI_rS1 = prI_rS1/sum(prI_rS1)
    prC_rS2 = prC_rS2/sum(prC_rS2)
    prI_rS2 = prI_rS2/sum(prI_rS2)

    # Sample 4 response classes from multinomial distirbution (normalised
    # within each response class)
    nC_rS1 = np.random.multinomial(CR, prC_rS1)
    nI_rS1 = np.random.multinomial(M, prI_rS1)
    nC_rS2 = np.random.multinomial(H, prC_rS2)
    nI_rS2 = np.random.multinomial(FA, prI_rS2)

    # Add to data vectors
    nR_S1 = np.hstack((nC_rS1, nI_rS2))
    nR_S2 = np.hstack((nI_rS1, nC_rS2))

    if as_df is True:
        return ratings2df(nR_S1, nR_S2)
    else:
        return nR_S1, nR_S2


def type2_SDT_simuation(d=1, noise=0.2, c=0, nRatings=4, nTrials=500):
    """Type 2 SDT simulation with variable noise.

    Parameters
    ----------
    d : float
        Type 1 dprime.
    noise : float or list
        Standard deviation of noise to be added to type 1 internal response for
        type 2 judgment. If noise is a 1 x 2 vector then this will simulate
        response-conditional type 2 data where
        `noise = [sigma_rS1, sigma_rS2]`.
    c : float
        Type 1 criterion.
    c1 : float
        Type 2 criteria for S1 response.
    c2 : float
        Type 2 criteria for S2 response.
    nRatings : int
        Number of ratings.
    nTrials : int
        Number of trials to simulate.

    Returns
    -------
    nR_S1, nR_S2 : 1d array-like
        nR_S1 and nR_S2 response counts.

    Examples
    --------

    """
    # Specify the confidence criterions based on the number of ratings
    c1 = c + np.linspace(-1.5, -0.5, (nRatings - 1))
    c2 = c + np.linspace(0.5, 1.5, (nRatings - 1))

    if isinstance(noise, list):
        rc = 1
        sigma1 = noise[0]
        sigma2 = noise[1]
    else:
        rc = 0
        sigma = noise

    S1mu = -d/2
    S2mu = d/2

    # Initialise response arrays
    nC_rS1 = np.zeros(len(c1)+1)
    nI_rS1 = np.zeros(len(c1)+1)
    nC_rS2 = np.zeros(len(c2)+1)
    nI_rS2 = np.zeros(len(c2)+1)

    for t in range(nTrials):

        s = round(np.random.rand())

        # Type 1 SDT model
        x = np.random.normal(S2mu, 1) if s == 1 else np.random.normal(S1mu, 1)

        # Add type 2 noise to signal
        if rc:  # add response-conditional noise
            if x < c:
                x2 = np.random.normal(x, sigma1) if sigma1 > 0 else x
            else:
                x2 = np.random.normal(x, sigma2) if sigma2 > 0 else x
        else:
            x2 = np.random.normal(x, sigma) if sigma > 0 else x

        # Generate confidence ratings
        if (s == 0) & (x < c):  # stimulus S1 and response S1
            i = np.where(np.hstack((c1, c)) >= x2)[0]
            if len(i) > 0:
                nC_rS1[i.min()] += 1

        elif (s == 0) & (x >= c):  # stimulus S1 and response S2
            i = np.where(np.hstack((c, c2)) <= x2)[0]
            if len(i) > 0:
                nI_rS2[i.max()] += 1

        elif (s == 1) & (x < c):  # stimulus S2 and response S1
            i = np.where(np.hstack((c1, c)) >= x2)[0]
            if len(i) > 0:
                nI_rS1[i.min()] += 1

        elif (s == 1) & (x >= c):  # stimulus S2 and response S2
            i = np.where(np.hstack((c, c2)) <= x2)[0]
            if len(i) > 0:
                nC_rS2[i.max()] += 1

    # Add to data vectors
    nR_S1 = np.hstack((nC_rS1, nI_rS2))
    nR_S2 = np.hstack((nI_rS1, nC_rS2))

    return nR_S1, nR_S2


def ratings2df(nR_S1, nR_S2):
    """Convert response count to dataframe.

    Parameters
    ----------
    nR_S1 : 1d array-like, list or string
        Confience ratings (stimuli 1, correct and incorrect).
    nR_S2 : 1d array-like, list or string
        Confience ratings (stimuli 2, correct and incorrect).

    Returns
    -------
    df : :py:class:`pandas.DataFrame`
         A DataFrame (nRows==`nTrials`) containing the responses and
         confidence rating for one participant given `nR_s1` and `nR_S2`.

    See also
    --------
    responseSimulation
    """
    df = pd.DataFrame([])
    nRatings = int(len(nR_S1)/2)
    for i in range(nRatings):
        if nR_S1[i]:
            df = df.append(pd.concat(
                [pd.DataFrame({'Stimuli': '1', 'Accuracy': 1,
                 'Confidence': [i+1]})]*nR_S1[i]))
        if nR_S2[i]:
            df = df.append(pd.concat(
                [pd.DataFrame({'Stimuli': '2', 'Accuracy': 1,
                 'Confidence': [i+1]})]*nR_S2[i]))
        if nR_S1[nRatings+i]:
            df = df.append(pd.concat(
                [pd.DataFrame({'Stimuli': '1', 'Accuracy': 0,
                 'Confidence': [i+1]})]*nR_S1[nRatings+i]))
        if nR_S2[nRatings+i]:
            df = df.append(pd.concat(
                [pd.DataFrame({'Stimuli': '2', 'Accuracy': 0,
                 'Confidence': [i+1]})]*nR_S2[nRatings+i]))
    df['nTrial'] = np.arange(len(df))  # Add a column for trials

    # Shuffles rows before returning
    return df.sample(frac=1).reset_index(drop=True)