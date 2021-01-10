# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.optimize import SR1, Bounds, LinearConstraint, minimize
from scipy.stats import norm
from metadPy.utils import trials2counts
from typing import Optional, Tuple, List, Union, Callable, overload, Any


@overload
def scores(
    data: None,
    stimuli: Union[List[Any], np.ndarray],
    responses: Union[List[Any], np.ndarray],
) -> Tuple[int, int, int, int]:
    ...


@overload
def scores(
    data: pd.DataFrame,
    stimuli: str,
    responses: str,
) -> Tuple[int, int, int, int]:
    ...


@pf.register_dataframe_method
def scores(
    data=None,
    stimuli=None,
    responses=None,
):
    """Extract hits, misses, false alarms and correct rejection from stimuli
    and responses vectors.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame` or None
        Dataframe containing one `stimuli` and one `response` column.
    stimuli : str, 1d array-like or list
        If a string is provided, should be the name of the column used as
        `stimuli`. If a list or an array is provided, should contain the
        boolean vectors for `stimuli`.
    responses : str or 1d array-like
        If a string is provided, should be the name of the column used as
        `responses`. If a list or an array is provided, should contain the
        boolean vector for `responses`.

    Returns
    -------
    hits, misses, fas, crs : int
        Return the number of hits, misees, false alarms and correct rejections.

    Notes
    -----
    If a :py:class:`pandas.DataFrame` is provided, the function will search for
    a `stimuli`and a `responses` column by default if no other column names are
    provided.
    """
    # Formatting checks
    if data is None:
        if (
            isinstance(stimuli, (np.ndarray, np.generic))
            and isinstance(responses, (np.ndarray, np.generic))
            and (len(stimuli) == len(responses))
            and np.all([s in [0, 1] for s in stimuli])
            and np.all([s in [0, 1] for s in responses])
        ):

            data = pd.DataFrame({"stimuli": stimuli, "responses": responses})
            stimuli, responses = "stimuli", "responses"
        else:
            raise ValueError(
                (
                    "If no data is provided, `stimuli` and",
                    " `responses` should be two boolean vectors",
                    " with equal lengths.",
                )
            )

    # Set default columns names if data is a DataFrame
    if stimuli is None:
        stimuli = "Stimuli"
    if responses is None:
        responses = "Responses"

    # Extract hits, misses, false alarm and correct rejection
    hits = sum(data[stimuli] & data[responses])
    misses = sum(data[stimuli] & ~data[responses])
    fas = sum(~data[stimuli] & data[responses])
    crs = sum(~data[stimuli] & ~data[responses])

    return hits, misses, fas, crs


@overload
def rates(
    data: None,
    hits: None,
    misses: None,
    fas: None,
    crs: None,
    stimuli: Union[List[Any], Any],
    responses: Union[List[Any], Any],
    correction: bool = ...,
) -> Tuple[float, float]:
    ...


@overload
def rates(
    data: pd.DataFrame,
    hits: None,
    misses: None,
    fas: None,
    crs: None,
    stimuli: str,
    responses: str,
    correction: bool = ...,
) -> Tuple[float, float]:
    ...


@overload
def rates(
    data: None,
    hits: int,
    misses: int,
    fas: int,
    crs: int,
    stimuli: None,
    responses: None,
    correction: bool = ...,
) -> Tuple[float, float]:
    ...


@pf.register_dataframe_method
def rates(
    data=None,
    hits=None,
    misses=None,
    fas=None,
    crs=None,
    stimuli=None,
    responses=None,
    correction=True,
):
    """Calculate hit and false alarm rates.

    The values are automatically corrected to avoid d' infinity (see below).

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame` or None
        Dataframe containing one `stimuli` and one `response` column.
    stimuli : str, 1d array-like or list
        If a string is provided, should be the name of the column used as
        `stimuli`. If a list or an array is provided, should contain the
        boolean vectors for `stimuli`. If `None` and `data` is a
        :py:class:`pandas.DataFrame`, will be set to `Stimuli` by default.
    responses : str or 1d array-like
        If a string is provided, should be the name of the column used as
        `responses`. If a list or an array is provided, should contain the
        boolean vector for `responses`. If `None` and `data` is a
        :py:class:`pandas.DataFrame`, will be set to `Responses` by default.
    hits : int or None
        Hits.
    misses :  int or None
        Misses.
    fas : int or None
        False alarms.
    crs : int or None
        Correct rejections.
    correction : bool
        Avoid d' infinity by correcting false alarm and hit rate vaules
        if equal to 0 or 1 using half inverse or 1 - half inverse.
        Half inverses values are defined by:
            half_hit = 0.5 / (hits + misses)
            half_fa = 0.5 / (fas + crs)
        Default is set to `True` (use correction).

    Returns
    -------
    hit_rate: float
        Hit rate.
    fa_rate : float
        False alarm rate.

    Info
    ----
    Will return hits rate and false alarm rates. The hits rate is defined by:

    .. math:: hits rate = \\frac{\\hits}{s_{hits + misses}}

    The false alarms rate is defined by:

    .. math:: fals alarms rate = \\frac{\\fals alarms}{s_{fals alarms + correct rejections}}

    .. warning:: This function will correct false alarm rates and hits rates
        by default using a half inverse method to avoid `0` and `1` values,
        which can bias d' estimates. Use `corretion=False` to compute
        uncorrected hits and false alarm rates.

    See also
    --------
    dprime, criterion, scores

    References
    ----------
    Adapted from: https://lindeloev.net/calculating-d-in-python-and-php/
    """
    if isinstance(data, pd.DataFrame):
        if stimuli is None:
            stimuli = "Stimuli"
        if responses is None:
            responses = "Responses"
        hits, misses, fas, crs = data.scores(stimuli=stimuli, responses=responses)
    elif data is not None:
        raise ValueError("Parameter `data` is not a dataframe.")

    if all(p is None for p in [hits, misses, fas, crs]):
        raise ValueError("No variable provided.")

    # Calculate hit_rate
    hit_rate = hits / (hits + misses)

    # Calculate false alarm rate
    fa_rate = fas / (fas + crs)

    if correction is True:  # avoid d' infinity if fa or hit is in [0, 1]

        # Floors an ceilings are replaced with half inverse hits and fa
        half_hit = 0.5 / (hits + misses)
        half_fa = 0.5 / (fas + crs)

        if hit_rate == 1:
            hit_rate = 1 - half_hit
        if hit_rate == 0:
            hit_rate = half_hit

        if fa_rate == 1:
            fa_rate = 1 - half_fa
        if fa_rate == 0:
            fa_rate = half_fa

    return float(hit_rate), float(fa_rate)


@overload
def dprime(
    data: pd.DataFrame,
    stimuli: str,
    responses: str,
    hit_rate: None,
    fa_rate: None,
) -> float:
    ...


@overload
def dprime(
    data: None,
    stimuli: None,
    responses: None,
    hit_rate: float,
    fa_rate: float,
) -> float:
    ...


@overload
def dprime(
    data: None,
    stimuli: Union[list, np.ndarray],
    responses: Union[list, np.ndarray],
    hit_rate: None,
    fa_rate: None,
) -> float:
    ...


@pf.register_dataframe_method
def dprime(
    data=None,
    stimuli=None,
    responses=None,
    hit_rate=None,
    fa_rate=None,
):
    """Calculate d prime.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame` or None
        Dataframe. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    hit_rate : float
        Hit rate.
    fa_rate : float
        False alarm rate.
    stimuli : string
        Name of the column containing the stimuli. If `None` and `data` is a
        :py:class:`pandas.DataFrame`, will be set to `Stimuli` by default.
    responses : string
        Name of the column containing the responses. If `None` and `data` is a
        :py:class:`pandas.DataFrame`, will be set to `Responses` by default.

    Returns
    -------
    dprime : float
        The d' value.

    Notes
    -----
    The d’ is a measure of the ability to discriminate a signal from noise.
    """
    if isinstance(data, pd.DataFrame):
        if stimuli is None:
            stimuli = "Stimuli"
        if responses is None:
            responses = "Responses"
        if (not isinstance(stimuli, str)) | (not isinstance(responses, str)):
            raise ValueError(
                "Parameters `stimuli` and `responses` must be strings",
                "when `data` is a DataFrame.",
            )
        hits, misses, fas, crs = scores(data=data, stimuli=stimuli, responses=responses)

        hit_rate, fa_rate = rates(hits=hits, misses=misses, fas=fas, crs=crs)
    elif (data is not None) & (
        ~isinstance(hit_rate, float) | ~isinstance(fa_rate, float)
    ):
        raise ValueError("No variable provided.")

    return norm.ppf(hit_rate) - norm.ppf(fa_rate)


@overload
def criterion(
    data: pd.DataFrame,
    hit_rate: None,
    fa_rate: None,
    stimuli: str,
    responses: str,
    correction: bool = True,
) -> float:
    ...


@overload
def criterion(
    data: None,
    hit_rate: float,
    fa_rate: float,
    stimuli: None,
    responses: None,
    correction: bool = True,
) -> float:
    ...


@overload
def criterion(
    data: None,
    hit_rate: None,
    fa_rate: None,
    stimuli: Union[list, np.ndarray],
    responses: Union[list, np.ndarray],
    correction: bool = True,
) -> float:
    ...


@pf.register_dataframe_method
def criterion(
    data=None,
    stimuli="Stimuli",
    responses="Responses",
    hit_rate=None,
    fa_rate=None,
):
    """Response criterion.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame` or None
        Dataframe. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    hit_rate : float
        Hit rate.
    fa_rate : float
        False alarm rate.
    stimuli : string
        Name of the column containing the stimuli.
    responses : string
        Name of the column containing the responses.

    Returns
    -------
    dprime : float
        The d' value.
    """
    if data is not None:
        if isinstance(data, pd.DataFrame):
            hits, misses, fas, crs = scores(
                data=data, stimuli=stimuli, responses=responses
            )
            hit_rate, fa_rate = rates(hits=hits, misses=misses, fas=fas, crs=crs)
        else:
            raise ValueError("data should be a dataframe")
    return -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))


def fit_meta_d_logL(parameters: list, inputObj: list) -> float:
    """Returns negative log-likelihood of parameters given experimental data.

    Parameters
    ----------
    parameters : list
        parameters[0] = meta d'
        parameters[1:end] = type-2 criteria locations
    inputObj : list
        List containing the following variables when called from
        `:py:func:metadPy.sdt.metad`:
            * nR_S1
            * nR_S2
            * nRatings
            * d1
            * t1c1
            * s
            * constant_criterion
            * fncdf
            * fninv
    """
    meta_d1 = parameters[0]
    t2c1 = parameters[1:]
    nR_S1, nR_S2, nRatings, d1, t1c1, s, constant_criterion, fncdf, fninv = inputObj

    # define mean and SD of S1 and S2 distributions
    S1mu = -meta_d1 / 2
    S1sd = 1
    S2mu = meta_d1 / 2
    S2sd = S1sd / s

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
    nC_rS2 = [nR_S2[i + nRatings] for i in range(nRatings)]
    nI_rS2 = [nR_S1[i + nRatings] for i in range(nRatings)]

    # get type 2 probabilities
    C_area_rS1 = fncdf(t1c1, S1mu, S1sd)
    I_area_rS1 = fncdf(t1c1, S2mu, S2sd)

    C_area_rS2 = 1 - fncdf(t1c1, S2mu, S2sd)
    I_area_rS2 = 1 - fncdf(t1c1, S1mu, S1sd)

    t2c1x = [-np.inf]
    t2c1x.extend(t2c1[0 : (nRatings - 1)])
    t2c1x.append(t1c1)
    t2c1x.extend(t2c1[(nRatings - 1) :])
    t2c1x.append(np.inf)

    prC_rS1 = [
        (fncdf(t2c1x[i + 1], S1mu, S1sd) - fncdf(t2c1x[i], S1mu, S1sd)) / C_area_rS1
        for i in range(nRatings)
    ]
    prI_rS1 = [
        (fncdf(t2c1x[i + 1], S2mu, S2sd) - fncdf(t2c1x[i], S2mu, S2sd)) / I_area_rS1
        for i in range(nRatings)
    ]

    prC_rS2 = [
        (
            (1 - fncdf(t2c1x[nRatings + i], S2mu, S2sd))
            - (1 - fncdf(t2c1x[nRatings + i + 1], S2mu, S2sd))
        )
        / C_area_rS2
        for i in range(nRatings)
    ]
    prI_rS2 = [
        (
            (1 - fncdf(t2c1x[nRatings + i], S1mu, S1sd))
            - (1 - fncdf(t2c1x[nRatings + i + 1], S1mu, S1sd))
        )
        / I_area_rS2
        for i in range(nRatings)
    ]

    # calculate logL
    logL = np.sum(
        [
            nC_rS1[i] * np.log(prC_rS1[i])
            + nI_rS1[i] * np.log(prI_rS1[i])
            + nC_rS2[i] * np.log(prC_rS2[i])
            + nI_rS2[i] * np.log(prI_rS2[i])
            for i in range(nRatings)
        ]
    )

    if np.isinf(logL) or np.isnan(logL):
        logL = -1e300  # returning -inf may cause optimize.minimize() to fail
    return -logL


@overload
def metad(
    data: None,
    nRatings: int,
    nR_S1: None,
    nR_S2: None,
    stimuli: Union[list, np.array],
    accuracy: Union[list, np.array],
    confidence: Union[list, np.array],
    padAmount: Optional[float] = None,
    s: int = 1,
    padding: bool = True,
    collapse: Optional[int] = None,
    fncdf: Callable[..., float] = norm.cdf,
    fninv: Callable[..., float] = norm.ppf,
    verbose: int = 1,
    output_df: bool = False,
) -> Union[dict, pd.DataFrame]:
    ...


@overload
def metad(
    data: None,
    nRatings: int,
    nR_S1: Union[list, np.array],
    nR_S2: Union[list, np.array],
    stimuli: None,
    accuracy: None,
    confidence: None,
    padAmount: Optional[float] = None,
    s: int = 1,
    padding: bool = True,
    collapse: Optional[int] = None,
    fncdf: Callable[..., float] = norm.cdf,
    fninv: Callable[..., float] = norm.ppf,
    verbose: int = 1,
    output_df: bool = False,
) -> Union[dict, pd.DataFrame]:
    ...


@overload
def metad(
    data: pd.DataFrame,
    nRatings: int,
    nR_S1: None,
    nR_S2: None,
    stimuli: str = "Stimuli",
    accuracy: str = "Accuracy",
    confidence: str = "Confidence",
    padAmount: Optional[float] = None,
    s: int = 1,
    padding: bool = True,
    collapse: Optional[int] = None,
    fncdf: Callable[..., float] = norm.cdf,
    fninv: Callable[..., float] = norm.ppf,
    verbose: int = 1,
    output_df: bool = False,
) -> Union[dict, pd.DataFrame]:
    ...


@pf.register_dataframe_method
def metad(
    data=None,
    nRatings=None,
    stimuli=None,
    accuracy=None,
    confidence=None,
    padAmount=None,
    nR_S1=None,
    nR_S2=None,
    s=1,
    padding=True,
    collapse=None,
    fncdf=norm.cdf,
    fninv=norm.ppf,
    verbose=1,
    output_df=False,
):
    """Estimate meta-d' using maximum likelihood estimation (MLE).

    This function is adapted from the transcription of fit_meta_d_MLE.m
    (Maniscalco & Lau, 2012) by Alan Lee:
    http://www.columbia.edu/~bsm2105/type2sdt/.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame` or None
        Dataframe. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    nRatings : int
        Number of discrete ratings. If a continuous rating scale was used, and
        the number of unique ratings does not match `nRatings`, will convert to
        discrete ratings using :py:func:`metadPy.utils.discreteRatings`.
    stimuli : string
        Name of the column containing the stimuli.
    accuracy : string
        Name of the columns containing the accuracy.
    confidence : string
        Name of the column containing the confidence ratings.
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
    s : int
        Ratio of standard deviations for type 1 distributions as:
        `s = np.std(S1) / np.std(S2)`. If not specified, s is set to a default
        value of 1. For most purposes, it is recommended to set `s=1`. See
        http://www.columbia.edu/~bsm2105/type2sdt for further discussion.
    padding : boolean
        If `True`, a small value will be added to the counts to avoid problems
        during fit.
    padAmount : float
        The value to add to each response count if padding is set to 1.
        Default value is 1/(2*nRatings)
    collapse : int or None
        If an integer `N` is provided, will collpase ratings to avoid zeros by
        summing every `N` consecutive ratings. Default set to `None`.
    fncdf : func
        A function handle for the CDF of the type 1 distribution. If not
        specified, fncdf defaults to :py:func:`scipy.stats.norm.cdf()`.
    fninv : func
        A function handle for the inverse CDF of the type 1 distribution. If
        not specified, fninv defaults to :py:func:`scipy.stats.norm.ppf()`.
    verbose : {0, 1, 2}
        Level of algorithm’s verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.
            * 3 : display progress during iterations (more complete report).

    Returns
    -------
    fit : dict or :py:class:`pandas.DataFrame`
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

    .. warning:: If nR_S1 or nR_S2 contain zeros, this may interfere with
        estimation of meta-d'. Some options for dealing with response cell
        counts containing zeros are:

        * Add a small adjustment factor (e.g. `1/(len(nR_S1)`, to each input
            vector. This is a generalization of the correction for similar
            estimation issues of type 1 d' as recommended in [1]_. When using
            this correction method, it is recommended to add the adjustment
            factor to ALL data for all subjects, even for those subjects whose
            data is not in need of such correction, in order to avoid biases in
            the analysis (cf [2]_). Use `padding==True` to activate this
            correction.

        * Collapse across rating categories. e.g. if your data set has 4
            possible confidence ratings such that `len(nR_S1)==8`, defining new
            input vectors:

            >>> nR_S1 = nR_S1.reshape(int(len(nR_S1)/collapse), 2).sum(axis=1)

            This might be sufficient to eliminate zeros from the input without
            using an adjustment. Use `collapse=True` to activate this
            correction.

    If there are N ratings, then there will be N-1 type 2 hit rates and false
    alarm rates.

    Examples
    --------
    No correction
    >>> nR_S1 = [36, 24, 17, 20, 10, 12, 9, 2]
    >>> nR_S2 = [1, 4, 10, 11, 19, 18, 28, 39]
    >>> fit = fit_meta_d_MLE(nR_S1, nR_S2, padding=False)

    Correction by padding values
    >>> nR_S1 = [36, 24, 17, 20, 10, 12, 9, 2]
    >>> nR_S2 = [1, 4, 10, 11, 19, 18, 28, 39]
    >>> fit = fit_meta_d_MLE(nR_S1, nR_S2, padding=True)

    Correction by collapsing values
    >>> nR_S1 = [36, 24, 17, 20, 10, 12, 9, 2]
    >>> nR_S2 = [1, 4, 10, 11, 19, 18, 28, 39]
    >>> fit = fit_meta_d_MLE(nR_S1, nR_S2, collapse=2)

    References
    ----------
    ..[1] Hautus, M. J. (1995). Corrections for extreme proportions and their
      biasing effects on estimated values of d'. Behavior Research Methods,
    Instruments, & Computers, 27, 46-51.

    ..[2] Snodgrass, J. G., & Corwin, J. (1988). Pragmatics of measuring
      recognition memory: Applications to dementia and amnesia. Journal of
      Experimental Psychology: General, 117(1), 34–50.
      https://doi.org/10.1037/0096-3445.117.1.34
    """
    if isinstance(data, pd.DataFrame):
        if padAmount is None:
            padAmount = 1 / (2 * nRatings)
        nR_S1, nR_S2 = trials2counts(
            data=data,
            stimuli=stimuli,
            accuracy=accuracy,
            confidence=confidence,
            nRatings=nRatings,
            padding=padding,
            padAmount=padAmount,
        )
    if isinstance(nR_S1, list):
        nR_S1 = np.array(nR_S1)
    if isinstance(nR_S2, list):
        nR_S2 = np.array(nR_S2)
    if (len(nR_S1) % 2) != 0:
        raise ValueError("input arrays must have an even number of elements")
    if len(nR_S1) != len(nR_S2):
        raise ValueError("input arrays must have the same number of elements")
    if (padding is False) & (collapse is None):
        if any(np.array(nR_S1) == 0) or any(np.array(nR_S2) == 0):
            import warnings

            warnings.warn(
                (
                    "Your inputs contain zeros and is not corrected. "
                    " This may interfere with proper estimation of meta-d."
                    " See docstrings for more information."
                )
            )
    elif (padding is True) & (collapse is None):
        # A small padding is required to avoid problems in model fit if any
        # confidence ratings aren't used (see Hautus MJ, 1995 for details)
        if padAmount is None:
            padAmount = 1 / len(nR_S1)
        nR_S1 = nR_S1 + padAmount
        nR_S2 = nR_S2 + padAmount
    elif (padding is False) & (collapse is not None):
        # Collapse values accross ratings to avoid problems in model fit
        nR_S1 = nR_S1.reshape(int(len(nR_S1) / collapse), 2).sum(axis=1)
        nR_S2 = nR_S2.reshape(int(len(nR_S2) / collapse), 2).sum(axis=1)
    elif (padding is True) & (collapse is not None):
        raise ValueError("Both padding and collapse are True.")

    nRatings = int(len(nR_S1) / 2)  # number of ratings in the experiment
    nCriteria = int(2 * nRatings - 1)  # number criteria to be fitted

    # parameters
    # meta-d' - 1
    # t2c - nCriteria-1
    # constrain type 2 criteria values, such that t2c(i) is always <= t2c(i+1)
    # -->  t2c(i+1) >= t2c(i) + 1e-5 (i.e. very small deviation from equality)
    A, ub, lb = [], [], []
    for ii in range(nCriteria - 2):
        tempArow = []
        tempArow.extend(np.zeros(ii + 1))
        tempArow.extend([1, -1])
        tempArow.extend(np.zeros((nCriteria - 2) - ii - 1))
        A.append(tempArow)
        ub.append(-1e-5)
        lb.append(-np.inf)

    # lower bounds on parameters
    LB = []
    LB.append(-10.0)  # meta-d'
    LB.extend(-20 * np.ones((nCriteria - 1) // 2))  # criteria lower than t1c
    LB.extend(np.zeros((nCriteria - 1) // 2))  # criteria higher than t1c

    # upper bounds on parameters
    UB = []
    UB.append(10.0)  # meta-d'
    UB.extend(np.zeros((nCriteria - 1) // 2))  # criteria lower than t1c
    UB.extend(20 * np.ones((nCriteria - 1) // 2))  # criteria higher than t1c

    # select constant criterion type
    constant_criterion = "meta_d1 * (t1c1 / d1)"  # relative criterion

    # set up initial guess at parameter values
    ratingHR = []
    ratingFAR = []
    for c in range(1, int(nRatings * 2)):
        ratingHR.append(sum(nR_S2[c:]) / sum(nR_S2))
        ratingFAR.append(sum(nR_S1[c:]) / sum(nR_S1))

    # obtain index in the criteria array to mark Type I and Type II criteria
    t1_index = nRatings - 1
    t2_index = list(set(list(range(0, 2 * nRatings - 1))) - set([t1_index]))

    d1 = (1 / s) * fninv(ratingHR[t1_index]) - fninv(ratingFAR[t1_index])
    meta_d1 = d1

    c1 = (-1 / (1 + s)) * (fninv(ratingHR) + fninv(ratingFAR))
    t1c1 = c1[t1_index]
    t2c1 = c1[t2_index]

    # initial values for the minimization function
    guess = [meta_d1]
    guess.extend(list(t2c1 - eval(constant_criterion)))

    # other inputs for the minimization function
    inputObj = [nR_S1, nR_S2, nRatings, d1, t1c1, s, constant_criterion, fncdf, fninv]
    bounds = Bounds(LB, UB)
    linear_constraint = LinearConstraint(A, lb, ub)

    # minimization of negative log-likelihood
    results = minimize(
        fit_meta_d_logL,
        guess,
        args=(inputObj),
        method="trust-constr",
        jac="2-point",
        hess=SR1(),
        constraints=[linear_constraint],
        options={"verbose": verbose},
        bounds=bounds,
    )

    # quickly process some of the output
    meta_d1 = results.x[0]
    t2c1 = results.x[1:] + eval(constant_criterion)
    logL = -results.fun

    # I_nR and C_nR are rating trial counts for incorrect and correct trials
    # element i corresponds to # (in)correct w/ rating i
    I_nR_rS2 = nR_S1[nRatings:]
    I_nR_rS1 = list(np.flip(nR_S2[0:nRatings], axis=0))

    C_nR_rS2 = nR_S2[nRatings:]
    C_nR_rS1 = list(np.flip(nR_S1[0:nRatings], axis=0))

    obs_FAR2_rS2 = [
        sum(I_nR_rS2[(i + 1) :]) / sum(I_nR_rS2) for i in range(nRatings - 1)
    ]
    obs_HR2_rS2 = [
        sum(C_nR_rS2[(i + 1) :]) / sum(C_nR_rS2) for i in range(nRatings - 1)
    ]
    obs_FAR2_rS1 = [
        sum(I_nR_rS1[(i + 1) :]) / sum(I_nR_rS1) for i in range(nRatings - 1)
    ]
    obs_HR2_rS1 = [
        sum(C_nR_rS1[(i + 1) :]) / sum(C_nR_rS1) for i in range(nRatings - 1)
    ]

    # find estimated t2FAR and t2HR
    S1mu = -meta_d1 / 2
    S1sd = 1
    S2mu = meta_d1 / 2
    S2sd = S1sd / s

    mt1c1 = eval(constant_criterion)

    C_area_rS2 = 1 - fncdf(mt1c1, S2mu, S2sd)
    I_area_rS2 = 1 - fncdf(mt1c1, S1mu, S1sd)

    C_area_rS1 = fncdf(mt1c1, S1mu, S1sd)
    I_area_rS1 = fncdf(mt1c1, S2mu, S2sd)

    est_FAR2_rS2, est_HR2_rS2 = [], []
    est_FAR2_rS1, est_HR2_rS1 = [], []

    for i in range(nRatings - 1):

        t2c1_lower = t2c1[(nRatings - 1) - (i + 1)]
        t2c1_upper = t2c1[(nRatings - 1) + i]

        I_FAR_area_rS2 = 1 - fncdf(t2c1_upper, S1mu, S1sd)
        C_HR_area_rS2 = 1 - fncdf(t2c1_upper, S2mu, S2sd)

        I_FAR_area_rS1 = fncdf(t2c1_lower, S2mu, S2sd)
        C_HR_area_rS1 = fncdf(t2c1_lower, S1mu, S1sd)

        est_FAR2_rS2.append(I_FAR_area_rS2 / I_area_rS2)
        est_HR2_rS2.append(C_HR_area_rS2 / C_area_rS2)

        est_FAR2_rS1.append(I_FAR_area_rS1 / I_area_rS1)
        est_HR2_rS1.append(C_HR_area_rS1 / C_area_rS1)

    # package output
    fit = {}
    fit["da"] = np.sqrt(2 / (1 + s ** 2)) * s * d1
    fit["s"] = s
    fit["meta_da"] = np.sqrt(2 / (1 + s ** 2)) * s * meta_d1
    fit["M_diff"] = fit["meta_da"] - fit["da"]
    fit["M_ratio"] = fit["meta_da"] / fit["da"]

    mt1c1 = eval(constant_criterion)
    fit["meta_ca"] = (np.sqrt(2) * s / np.sqrt(1 + s ** 2)) * mt1c1

    t2ca = (np.sqrt(2) * s / np.sqrt(1 + s ** 2)) * np.array(t2c1)
    fit["t2ca_rS1"] = t2ca[0 : nRatings - 1]
    fit["t2ca_rS2"] = t2ca[(nRatings - 1) :]

    fit["d1"] = d1
    fit["meta_d1"] = meta_d1
    fit["s"] = s
    fit["meta_c1"] = mt1c1
    fit["t2c1_rS1"] = t2c1[0 : nRatings - 1]
    fit["t2c1_rS2"] = t2c1[(nRatings - 1) :]
    fit["logL"] = logL

    fit["est_HR2_rS1"] = est_HR2_rS1
    fit["obs_HR2_rS1"] = obs_HR2_rS1

    fit["est_FAR2_rS1"] = est_FAR2_rS1
    fit["obs_FAR2_rS1"] = obs_FAR2_rS1

    fit["est_HR2_rS2"] = est_HR2_rS2
    fit["obs_HR2_rS2"] = obs_HR2_rS2

    fit["est_FAR2_rS2"] = est_FAR2_rS2
    fit["obs_FAR2_rS2"] = obs_FAR2_rS2

    if output_df is True:
        return pd.DataFrame(fit)
    elif output_df is False:
        return fit


def roc_auc(nR_S1: Union[list, np.array], nR_S2: Union[list, np.array]) -> float:
    """Calculate the area under the type 2 ROC curve given nR_S1 and nR_S2
    ratings counts.

    Parameters
    ----------
    nR_S1 : list or 1d array-like
        Confience ratings (stimuli 1).
    nR_S2 : list or 1d array-like
        Confidence ratings (stimuli 2).

    Returns
    -------
    auc : float
        Area under the type 2 ROC curve.

    Examples
    --------
    >>> nR_S1 = [36, 24, 17, 20, 10, 12, 9, 2]
    >>> nR_S2 = [1, 4, 10, 11, 19, 18, 28, 39]
    >>> roc_auc(nR_S1, nR_S2)
    0.6998064266356949
    """
    if isinstance(nR_S1, list):
        nR_S1 = np.array(nR_S1)
    if isinstance(nR_S2, list):
        nR_S2 = np.array(nR_S2)

    nRatings = int(len(nR_S1) / 2)

    flip_nR_S1 = np.flip(nR_S1)
    flip_nR_S2 = np.flip(nR_S2)

    S1_H2 = (nR_S1 + 0.5)[:nRatings]
    S2_H2 = (flip_nR_S2 + 0.5)[:nRatings]
    S1_FA2 = (flip_nR_S1 + 0.5)[:nRatings]
    S2_FA2 = (nR_S2 + 0.5)[:nRatings]

    H2 = S1_H2 + S2_H2
    FA2 = S1_FA2 + S2_FA2

    H2 /= sum(H2)
    FA2 /= sum(FA2)
    cum_H2 = np.hstack((0, np.cumsum(H2)))
    cum_FA2 = np.hstack((0, np.cumsum(FA2)))

    k = []
    for c in range(nRatings):
        k.append((cum_H2[c + 1] - cum_FA2[c]) ** 2 - (cum_H2[c] - cum_FA2[c + 1]) ** 2)

    return 0.5 + 0.25 * sum(k)
