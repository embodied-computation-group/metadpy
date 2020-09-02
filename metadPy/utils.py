# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>


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
