# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, List, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.stats import norm


@overload
def trials2counts(
    data: None,
    stimuli: Union[list, np.ndarray],
    responses: Union[list, np.ndarray],
    accuracy: Union[list, np.ndarray],
    confidence: Union[list, np.ndarray],
    nRatings: int = 4,
    padding: bool = False,
    padAmount: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ...


@overload
def trials2counts(
    data=pd.DataFrame,
    stimuli: str = "Stimuli",
    responses: str = "Responses",
    accuracy: str = "Accuracy",
    confidence: str = "Confidence",
    nRatings: int = 4,
    padding: bool = False,
    padAmount: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ...


@pf.register_dataframe_method
def trials2counts(
    data=None,
    stimuli="Stimuli",
    responses="Responses",
    accuracy="Accuracy",
    confidence="Confidence",
    nRatings=4,
    padding=False,
    padAmount=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw behavioral data to nR_S1 and nR_S2 response count.

    Given data from an experiment where an observer discriminates between two
    stimulus alternatives on every trial and provides confidence ratings,
    converts trial by trial experimental information for N trials into response
    counts.

    Parameters
    ----------
    data :
        Dataframe containing stimuli, accuracy and confidence ratings.
    stimuli :
        Stimuli ID (0 or 1). If a dataframe is provided, should be the name of
        the column containing the stimuli ID. Default is `'Stimuli'`.
    responses :
        Response (0 or 1). If a dataframe is provided, should be the
        name of the column containing the response accuracy. Default is
        `'Responses'`.
    accuracy :
        Response accuracy (0 or 1). If a dataframe is provided, should be the
        name of the column containing the response accuracy. Default is
        `'Accuracy'`.
    confidence :
        Confidence ratings. If a dataframe is provided, should be the name of
        the column containing the confidence ratings. Default is
        `'Confidence'`.
    nRatings :
        Total of available subjective ratings available for the subject. e.g.
        if subject can rate confidence on a scale of 1-4, then nRatings = 4.
        Default is `4`.
    padding :
        If `True`, each response count in the output has the value of padAmount
        added to it. Padding cells is desirable if trial counts of 0 interfere
        with model fitting. If False, trial counts are not manipulated and 0s
        may be present in the response count output. Default value for padding
        is 0.
    padAmount :
        The value to add to each response count if padding is set to 1.
        Default value is 1/(2*nRatings)

    Returns
    -------
    nR_S1, nR_S2 :
        Vectors containing the total number of responses in each accuracy
        category, conditional on presentation of S1 and S2.

    Notes
    -----
    All trials where `stimuli` is not 0 or 1, accuracy is not 0 or 1, or confidence is
    not in the range [1, nRatings], are automatically omitted.

    The inputs can be responses, accuracy or both. If both `responses` and
    `accuracy` are provided, will check for consstency. If only `accuracy` is
    provided, the responses vector will be automatically infered.

    If nR_S1 = [100 50 20 10 5 1], then when stimulus S1 was presented, the subject had
    the following accuracy counts:
        responded S1, confidence=3 : 100 times
        responded S1, confidence=2 : 50 times
        responded S1, confidence=1 : 20 times
        responded S2, confidence=1 : 10 times
        responded S2, confidence=2 : 5 times
        responded S2, confidence=3 : 1 time

    The ordering of accuracy / confidence counts for S2 should be the same as it is for
    S1. e.g. if nR_S2 = [3 7 8 12 27 89], then when stimulus S2 was presented, the
    subject had the following accuracy counts:
        responded S1, confidence=3 : 3 times
        responded S1, confidence=2 : 7 times
        responded S1, confidence=1 : 8 times
        responded S2, confidence=1 : 12 times
        responded S2, confidence=2 : 27 times
        responded S2, confidence=3 : 89 times

    Examples
    --------
    >>> stimID = [0, 1, 0, 0, 1, 1, 1, 1]
    >>> accuracy = [0, 1, 1, 1, 0, 0, 1, 1]
    >>> confidence = [1, 2, 3, 4, 4, 3, 2, 1]
    >>> nRatings = 4

    >>> nR_S1, nR_S2 = trials2counts(stimID, accuracy, confidence, nRatings)
    >>> print(nR_S1, nR_S2)

    Reference
    ---------
    This function is adapted from the Python version of trials2counts.m by
    Maniscalco & Lau [1] retrieved at:
    http://www.columbia.edu/~bsm2105/type2sdt/trials2counts.py

    .. [1] Maniscalco, B., & Lau, H. (2012). A signal detection theoretic
        approach for estimating metacognitive sensitivity from confidence
        ratings. Consciousness and Cognition, 21(1), 422–430.
        https://doi.org/10.1016/j.concog.2011.09.021

    """
    if isinstance(data, pd.DataFrame):
        stimuli = data[stimuli].to_numpy()
        confidence = data[confidence].to_numpy()
        if accuracy in data:
            accuracy = data[accuracy].to_numpy()
        if responses in data:
            responses = data[responses].to_numpy()
    elif data is not None:
        raise ValueError("`Data` should be a DataFrame")

    if isinstance(accuracy, str) & isinstance(responses, str):
        raise ValueError("Neither `responses` nor `accuracy` are provided")

    # Create responses vector if missing
    if isinstance(responses, str):
        responses = stimuli.copy()
        responses[accuracy == 0] = 1 - responses[accuracy == 0]

    # Check for valid inputs
    if not np.all(np.array([len(responses), len(confidence)]) == len(stimuli)):
        raise ValueError("Input vectors must have the same length")

    # Check data consistency
    tempstim, tempresp, tempratg = [], [], []
    for s, rp, rt in zip(stimuli, responses, confidence):
        if (s == 0 or s == 1) and (rp == 0 or rp == 1) and (rt >= 1 and rt <= nRatings):
            tempstim.append(s)
            tempresp.append(rp)
            tempratg.append(rt)
    stimuli = tempstim
    responses = tempresp
    confidence = tempratg

    if padAmount is None:
        padAmount = 1 / (2 * nRatings)

    nR_S1, nR_S2 = [], []
    # S1 responses
    for r in range(nRatings, 0, -1):
        cs1, cs2 = 0, 0
        for s, rp, rt in zip(stimuli, responses, confidence):
            if s == 0 and rp == 0 and rt == r:
                cs1 += 1
            if s == 1 and rp == 0 and rt == r:
                cs2 += 1
        nR_S1.append(cs1)
        nR_S2.append(cs2)

    # S2 responses
    for r in range(1, nRatings + 1, 1):
        cs1, cs2 = 0, 0
        for s, rp, rt in zip(stimuli, responses, confidence):
            if s == 0 and rp == 1 and rt == r:
                cs1 += 1
            if s == 1 and rp == 1 and rt == r:
                cs2 += 1
        nR_S1.append(cs1)
        nR_S2.append(cs2)

    # pad response counts to avoid zeros
    if padding:
        nR_S1 = [n + padAmount for n in nR_S1]
        nR_S2 = [n + padAmount for n in nR_S2]

    return np.array(nR_S1), np.array(nR_S2)


def discreteRatings(
    ratings: Union[list, np.ndarray],
    nbins: int = 4,
    verbose: bool = True,
    ignore_invalid: bool = False,
) -> Tuple[np.ndarray, Dict[str, list]]:
    """Convert from continuous to discrete ratings.

    Resample if quantiles are equal at high or low end to ensure proper
    assignment of binned confidence

    Parameters
    ----------
    ratings : list | np.ndarray
        Ratings on a continuous scale.
    nbins : int
        The number of discrete ratings to resample. Defaut set to `4`.
    verbose : boolean
        If `True`, warning warnings be returned.
    ignore_invalid : bool
        If `False` (default), an arreor will be raised in case of impossible
        discretisation of the confidence ratings. This is mostly due to identical
        values and SDT values should not be extracted from the data. If `True` the
        discretisation will process anyway. This option can be usefull for plotting.

    Returns
    -------
    discreteRatings : np.ndarray
        New rating array only containing integers between 1 and `nbins`.
    out : dict
        Dictionary containing logs of the discrization process:
            * `'confbins'`: list or 1d array-like - If the ratings were
                reampled, a list containing the new ratings and the new low or
                hg threshold, appened before or after the rating, respectively.
                Else, only returns the ratings.
            * `'rebin'`: boolean - If True, the ratings were resampled due to
                larger numbers of highs or low ratings.
            * `'binCount'` : int - Number of bins

    .. warning:: This function will automatically control for bias in high or
        low confidence ratings. If the first two or the last two quantiles
        have identical values, low or high confidence trials are excluded
        (respectively), and the function is run again on the remaining data.

    Raises
    ------
    ValueError:
        If the confidence ratings contains a lot of identical values and
        `ignore_invalid` is `False`.

    Examples
    --------
    >>> from metadpy.utils import discreteRatings
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
    confBins = np.quantile(ratings, np.linspace(0, 1, nbins + 1))
    if (confBins[0] == confBins[1]) & (confBins[nbins - 1] == confBins[nbins]):
        if ignore_invalid is False:
            raise ValueError(
                "The resulting rating scale contains a lot of identical"
                " values and cannot be further analyzed"
            )
    elif confBins[nbins - 1] == confBins[nbins]:
        if verbose is True:
            print("Correcting for bias in high confidence ratings")
        # Exclude high confidence trials and re-estimate
        hiConf = confBins[-1]
        confBins = np.quantile(ratings[ratings != hiConf], np.linspace(0, 1, nbins))
        for b in range(len(confBins) - 1):
            temp.append((ratings >= confBins[b]) & (ratings <= confBins[b + 1]))
        temp.append(ratings == hiConf)

        out["confBins"] = [confBins, hiConf]
        out["rebin"] = [1]
    elif confBins[0] == confBins[1]:
        if verbose is True:
            print("Correction for bias in low confidence ratings")
        # Exclude low confidence trials and re-estimate
        lowConf = confBins[1]
        temp.append(ratings == lowConf)
        confBins = np.quantile(ratings[ratings != lowConf], np.linspace(0, 1, nbins))
        for b in range(1, len(confBins)):
            temp.append((ratings >= confBins[b - 1]) & (ratings <= confBins[b]))
        out["confBins"] = [lowConf, confBins]
        out["rebin"] = [1]
    else:
        for b in range(len(confBins) - 1):
            temp.append((ratings >= confBins[b]) & (ratings <= confBins[b + 1]))
        out["confBins"] = confBins
        out["rebin"] = [0]

    discreteRatings = np.zeros(len(ratings), dtype="int")
    for b in range(nbins):
        discreteRatings[temp[b]] = b
    discreteRatings += 1
    out["binCount"] = [sum(temp[b])]

    return discreteRatings, out


def trialSimulation(
    d: float = 1.0,
    metad: float = 2.0,
    mRatio: float = 1,
    c: float = 0,
    nRatings: int = 4,
    nTrials: int = 500,
) -> pd.DataFrame:
    """Simulate nR_S1 and nR_S2 response counts.

    Parameters
    ----------
    d : float
        Type 1 task performance (d prime).
    metad : float
        Type 2 sensitivity in units of type 1 dprime.
    mRatio : float
        Specify Mratio (meta-d/d'). If `len(mRatio)>1`, mRatios are assumed to be
        drawn from a repeated measures design.
    c : float
        Type 1 task bias (criterion).
    nRatings : int
        Number of ratings.
    nTrials : int
        Set the number of trials performed.

    Returns
    -------
    output_df : :py:class:`pandas.DataFrame`
        A DataFrame (nRows==`nTrials`) containing the responses and confidence rating
        for one participant given the provided parameters.

    References
    ----------
    This function is adapted from the Matlab `cpc_metad_sim` function from:
    https://github.com/metacoglab/HMeta-d/blob/master/CPC_metacog_tutorial/cpc_metacog_utils/cpc_metad_sim.m

    See Also
    --------
    ratings2df

    """
    # Specify the confidence criterions based on the number of ratings
    c1 = c + np.linspace(-1.5, -0.5, (nRatings - 1))
    c2 = c + np.linspace(0.5, 1.5, (nRatings - 1))

    # Calc type 1 response counts
    H = round((1 - norm.cdf(c, d / 2)) * (nTrials / 2))
    FA = round((1 - norm.cdf(c, -d / 2)) * (nTrials / 2))
    CR = round(norm.cdf(c, -d / 2) * (nTrials / 2))
    M = round(norm.cdf(c, d / 2) * (nTrials / 2))

    # Calc type 2 probabilities
    S1mu = -metad / 2
    S2mu = metad / 2

    # Normalising constants
    C_area_rS1 = norm.cdf(c, S1mu)
    I_area_rS1 = norm.cdf(c, S2mu)
    C_area_rS2 = 1 - norm.cdf(c, S2mu)
    I_area_rS2 = 1 - norm.cdf(c, S1mu)

    t2c1x = np.hstack((-np.inf, c1, c, c2, np.inf))

    prC_rS1, prI_rS1, prC_rS2, prI_rS2 = [], [], [], []
    for i in range(nRatings):
        prC_rS1.append(
            (norm.cdf(t2c1x[i + 1], S1mu) - norm.cdf(t2c1x[i], S1mu)) / C_area_rS1
        )
        prI_rS1.append(
            (norm.cdf(t2c1x[i + 1], S2mu) - norm.cdf(t2c1x[i], S2mu)) / I_area_rS1
        )
        prC_rS2.append(
            (
                (1 - norm.cdf(t2c1x[nRatings + i], S2mu))
                - (1 - norm.cdf(t2c1x[nRatings + i + 1], S2mu))
            )
            / C_area_rS2
        )
        prI_rS2.append(
            (
                (1 - norm.cdf(t2c1x[nRatings + i], S1mu))
                - (1 - norm.cdf(t2c1x[nRatings + i + 1], S1mu))
            )
            / I_area_rS2
        )

    # Ensure vectors sum to 1 to avoid problems with mnrnd
    prC_rS1 = np.array(prC_rS1) / sum(np.array(prC_rS1))  # type: ignore
    prI_rS1 = np.array(prI_rS1) / sum(np.array(prI_rS1))  # type: ignore
    prC_rS2 = np.array(prC_rS2) / sum(np.array(prC_rS2))  # type: ignore
    prI_rS2 = np.array(prI_rS2) / sum(np.array(prI_rS2))  # type: ignore

    # Sample 4 response classes from multinomial distirbution (normalized
    # within each response class)
    nC_rS1 = np.random.multinomial(CR, prC_rS1)
    nI_rS1 = np.random.multinomial(M, prI_rS1)
    nC_rS2 = np.random.multinomial(H, prC_rS2)
    nI_rS2 = np.random.multinomial(FA, prI_rS2)

    # Add to data vectors
    nR_S1 = np.hstack((nC_rS1, nI_rS2))
    nR_S2 = np.hstack((nI_rS1, nC_rS2))

    output_df = ratings2df(nR_S1, nR_S2)

    return output_df


def responseSimulation(
    d: float = 1.0,
    metad: float = 2.0,
    c: float = 0,
    nRatings: int = 4,
    nTrials: int = 500,
    nSubjects: int = 1,
) -> pd.DataFrame:
    """Simulate response and confidence ratings for one or a group of participants.

    Parameters
    ----------
    d :
        Type 1 task performance (d prime).
    metad : float
        Type 2 sensitivity in units of type 1 dprime.
    c :
        Type 1 task bias (criterion).
    nRatings :
        Number of ratings.
    nTrials :
        Set the number of trials performed.
    nSubjects :
        Specify the number of subject who performed the task.

    Returns
    -------
    output_df :
        A DataFrame (nRows==`nTrials`) containing the responses and
        confidence rating for one or many participants given the provided
        parameters.

    References
    ----------
    This function is adapted from the Matlab `cpc_metad_sim` function from:
    https://github.com/metacoglab/HMeta-d/blob/master/CPC_metacog_tutorial/cpc_metacog_utils/cpc_metad_sim.m

    See Also
    --------
    ratings2df

    """
    output_df = pd.DataFrame([])
    for sub in range(nSubjects):
        this_df = trialSimulation(
            d=d,
            metad=metad,
            c=c,
            nRatings=nRatings,
            nTrials=nTrials,
        )
        this_df["Subject"] = sub
        output_df = pd.concat([output_df, this_df], ignore_index=True)

    return output_df


def pairedResponseSimulation(
    d: float = 1.0,
    d_sigma: float = 0.1,
    mRatio: list = [1, 0.6],
    mRatio_sigma: float = 0.2,
    mRatio_rho: float = 0,
    c: float = 0,
    c_sigma: float = 0.1,
    nRatings: int = 4,
    nTrials: int = 500,
    nSubjects: int = 20,
) -> pd.DataFrame:
    """Simulate response and confidence ratings a group with 2 experimental conditions.

    Parameters
    ----------
    d :
        Type 1 task performance (d prime).
    d_sigma :
        Include some between-subject variability for d prime.
    mRatio :
        Specify Mratio (meta-d/d'). If `len(mRatio)>1`, mRatios are assumed to be drawn
        from a repeated measures design.
    mRatio_sigma :
        Include some variability in the mRatio scores.
    mRatio_rho :
        Specify the correlation between the two Mratios.
    c :
        Type 1 task bias (criterion).
    c_sigma :
        Include some between-subject variability for criterion.
    nRatings :
        Number of ratings.
    nTrials :
        Set the number of trials performed.
    nSubjects :
        Specify the number of subject who performed the task. Defaults to `20`.

    Returns
    -------
    output_df :
        A DataFrame (nRows==`nTrials` * `nSubjects`) containing the
        responses and confidence rating for one or many participants
        given the provided parameters.

    References
    ----------
    This function is adapted from the Matlab `cpc_metad_sim` function from:
    https://github.com/metacoglab/HMeta-d/blob/master/CPC_metacog_tutorial/cpc_metacog_utils/cpc_metad_sim.m

    See Also
    --------
    ratings2df

    """
    # Create covariance matrix for the two mRatios
    covMatrix = np.array(
        [
            [mRatio_sigma**2, mRatio_rho * mRatio_sigma**2],
            [mRatio_rho * mRatio_sigma**2, mRatio_sigma**2],
        ]
    )
    MVvalues = np.zeros((nSubjects, 2))
    d_vector, c_vector = np.zeros((nSubjects, 2)), np.zeros((nSubjects, 2))
    metad_list = np.zeros((nSubjects, 2))
    output_df = pd.DataFrame([])

    for b in range(nSubjects):
        # Generate the Mratio values
        # from a multivariate normal distribution
        MVvalues[b, :] = np.random.multivariate_normal(mRatio, covMatrix)
        for a in range(2):
            # Generate dprime values
            d_vector[b, a] = np.random.normal(d, d_sigma)
            # Generate bias values
            c_vector[b, a] = np.random.normal(c, c_sigma)
            # Generate meta-d values
            metad_list[b, a] = MVvalues[b, a] * d_vector[b, a]
            # Simulate data
            this_df = trialSimulation(
                d=d_vector[b, a],
                metad=metad_list[b, a],
                c=c_vector[b, a],
                nRatings=nRatings,
                nTrials=nTrials,
            )
            this_df["Subject"] = b
            this_df["Condition"] = a
            output_df = pd.concat([output_df, this_df], ignore_index=True)

    return output_df


def type2_SDT_simuation(
    d: float = 1,
    noise: Union[float, List[float]] = 0.2,
    c: float = 0,
    nRatings: int = 4,
    nTrials: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Type 2 SDT simulation with variable noise.

    Parameters
    ----------
    d :
        Type 1 dprime.
    noise :
        Standard deviation of noise to be added to type 1 internal response for
        type 2 judgment. If noise is a 1 x 2 vector then this will simulate
        response-conditional type 2 data where
        `noise = [sigma_rS1, sigma_rS2]`.
    c :
        Type 1 criterion.
    c1 :
        Type 2 criteria for S1 response.
    c2 :
        Type 2 criteria for S2 response.
    nRatings :
        Number of ratings.
    nTrials :
        Number of trials to simulate.

    Returns
    -------
    nR_S1, nR_S2 : 1d array-like
        nR_S1 and nR_S2 response counts.

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

    S1mu = -d / 2
    S2mu = d / 2

    # Initialise response arrays
    nC_rS1 = np.zeros(len(c1) + 1)
    nI_rS1 = np.zeros(len(c1) + 1)
    nC_rS2 = np.zeros(len(c2) + 1)
    nI_rS2 = np.zeros(len(c2) + 1)

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


def ratings2df(nR_S1: np.ndarray, nR_S2: np.ndarray) -> pd.DataFrame:
    """Convert response count to dataframe.

    Parameters
    ----------
    nR_S1 :
        Confience ratings (stimuli 1, correct and incorrect).
    nR_S2 :
        Confience ratings (stimuli 2, correct and incorrect).

    Returns
    -------
    df :
         A DataFrame (nRows==`nTrials`) containing the responses, accuracy and
         confidence rating for one participant given `nR_s1` and `nR_S2`.

    See Also
    --------
    responseSimulation, trials2counts

    """
    df = pd.DataFrame([])
    nRatings = int(len(nR_S1) / 2)
    for i in range(nRatings):
        if nR_S1[i]:
            df = pd.concat(
                [
                    df,
                    pd.concat(
                        [
                            pd.DataFrame(
                                {
                                    "Stimuli": 0,
                                    "Responses": 0,
                                    "Accuracy": 1,
                                    "Confidence": [nRatings - i],
                                }
                            )
                        ]
                        * nR_S1[i]
                    ),
                ]
            )
        if nR_S2[i]:
            df = pd.concat(
                [
                    df,
                    pd.concat(
                        [
                            pd.DataFrame(
                                {
                                    "Stimuli": 1,
                                    "Responses": 0,
                                    "Accuracy": 0,
                                    "Confidence": [nRatings - i],
                                }
                            )
                        ]
                        * nR_S2[i]
                    ),
                ]
            )
        if nR_S1[nRatings + i]:
            df = pd.concat(
                [
                    df,
                    pd.concat(
                        [
                            pd.DataFrame(
                                {
                                    "Stimuli": 0,
                                    "Responses": 1,
                                    "Accuracy": 0,
                                    "Confidence": [i + 1],
                                }
                            )
                        ]
                        * nR_S1[nRatings + i]
                    ),
                ]
            )
        if nR_S2[nRatings + i]:
            df = pd.concat(
                [
                    df,
                    pd.concat(
                        [
                            pd.DataFrame(
                                {
                                    "Stimuli": 1,
                                    "Responses": 1,
                                    "Accuracy": 1,
                                    "Confidence": [i + 1],
                                }
                            )
                        ]
                        * nR_S2[nRatings + i]
                    ),
                ]
            )

    # Shuffles rows before returning
    df = df.sample(frac=1).reset_index(drop=True)
    df["nTrial"] = np.arange(len(df))  # Add a column for trials number

    return df
