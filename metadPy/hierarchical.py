# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from metadPy.sdt import dprime, criterion
from metadPy.utils import discreteRatings, trials2counts
import numpy as np


def hmetad(data, nR_S1=None, nR_S2=None, stimuli=None, accuracy=None,
           confidence=None, nRatings=None, within=None, between=None,
           subject=None, nbins=4):
    """Estimate parameters of the Hierarchical Bayesian meta-d'

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Dataframe. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    nR_S1 : 1d array-like, list or string
        Confience ratings (stimuli 1, correct and incorrect).
    nR_S2 : 1d array-like, list or string
        Confience ratings (stimuli 2, correct and incorrect).
    stimuli : string
        Name of the column containing the stimuli.
    accuracy : string
        Name of the columns containing the accuracy.
    confidence : string
        Name of the column containing the confidence ratings.
    nRatings : int
        Number of discrete ratings. If a continuous rating scale was used, and
        the number of unique ratings does not match `nRatings`, will convert to
        discrete ratings using :py:func:`metadPy.utils.discreteRatings`.
    within : string
        Name of column containing the within factor (condition comparison).
    between : string
        Name of column containing the between subject factor (group
        comparison).
    subject : string
        Name of column containing the subject identifier (only required if a
        within-subject or a between-subject factor is provided).
    nbins : int
        If a continuous rating scale was using, `nbins` define the number of
        discrete ratings when converting using
        :py:func:`metadPy.utils.discreteRatings`. The default value is `4`.

    Returns
    -------
    model : dict
        The fitted model.

    Examples
    --------
    1. Subject-level

    2. Group-level

    3. Repeated measures

    References
    ----------
    .. [#] Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation of
    metacognitive efficiency from confidence ratings, Neuroscience of
    Consciousness, 3(1) nix007, https://doi.org/10.1093/nc/nix007
    """
    # If a continuous rating scale was used (if N unique ratings > nRatings)
    # transform confidence to discrete ratings
    if data[confidence].nunique() > nRatings:
        data[confidence] = discreteRatings(data[confidence].to_numpy(),
                                           nbins=nbins)

    nR_S1, nR_S2 = trials2counts(
        data=data, stimuli=stimuli, accuracy=accuracy, confidence=confidence,
        nRatings=nRatings)

    data = preprocess(np.asarray(nR_S1), np.asarray(nR_S2))

    from models import hmetad_subjectLevel
    traces = hmetad_subjectLevel(data, chains=3, tune=1000, draws=1000)

    return traces


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
