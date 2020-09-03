# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np

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

    response : list or 1d array-like

    rating : list or 1d array-like

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
    if not (len(stimID) == len(response)) and (len(stimID) == len(rating)):
        raise('Input vectors must have the same lengths')

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
