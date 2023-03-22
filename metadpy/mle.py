# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import sys
import warnings
from math import sqrt
from typing import Callable, Dict, List, Optional, Union, overload

import numpy as np
import pandas as pd
import pandas_flavor as pf
import scipy.special.cython_special as cysp
from numba import jit, vectorize
from numba.extending import get_cython_function_address
from numba.types import WrapperAddressProtocol, float32, float64
from scipy.optimize import SR1, Bounds, LinearConstraint, minimize
from scipy.stats import norm
from tqdm import tqdm

from metadpy.utils import trials2counts

warnings.filterwarnings(
    "ignore", message="delta_grad == 0.0. Check if the approximated function is linear."
)

_signatures = [
    float32(float32, float32, float32),
    float64(float64, float64, float64),
]


def get(name, signature):
    """Get Scipy Cython function."""
    index = 1 if signature.return_type is float64 else 0
    pyx_fuse_name = f"__pyx_fuse_{index}{name}"
    if pyx_fuse_name in cysp.__pyx_capi__:
        name = pyx_fuse_name
    addr = get_cython_function_address("scipy.special.cython_special", name)

    cls = type(
        name,
        (WrapperAddressProtocol,),
        {"__wrapper_address__": lambda self: addr, "signature": lambda self: signature},
    )
    return cls()


erf = get("erf", float64(float64))


@vectorize(_signatures)
def norm_cdf(
    x: Union[float, np.ndarray],
    mu: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
):
    """Evaluate cumulative distribution function of normal distribution."""
    z = (x - mu) / sigma
    z *= 1.0 / sqrt(2)
    return 0.5 * (1.0 + erf(z))


@jit(nopython=True)
def fit_meta_d_logL(
    guess: np.ndarray,
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
    nRatings: int,
    d1: float,
    t1c1: float,
    s: int,
) -> float:
    """Negative log-likelihood of parameters given experimental data.

    Parameters
    ----------
    guess :
        guess[0] = meta d'
        guess[1:end] = type-2 criteria locations
    nR_S1, nR_S2 :
        These are vectors containing the total number of responses in each response
        category, conditional on presentation of S1 and S2.
    nRatings :
        Numbers of ratings.
    d1 :
        d prime.
    t1c1 :
        The type-1 criterion.
    s :
        Ratio of standard deviations for type 1 distributions as:
        `s = np.std(S1) / np.std(S2)`. If not specified, s is set to a default
        value of 1. For most purposes, it is recommended to set `s=1`. See
        http://www.columbia.edu/~bsm2105/type2sdt for further discussion.

    """
    meta_d1, t2c1 = guess[0], guess[1:]

    # define mean and SD of S1 and S2 distributions
    S1mu = -meta_d1 / 2
    S1sd = 1
    S2mu = meta_d1 / 2
    S2sd = S1sd / s

    # adjust so that the type 1 criterion is set at 0
    # (this is just to work with optimization toolbox constraints...
    #  to simplify defining the upper and lower bounds of type 2 criteria)
    S1mu = S1mu - (meta_d1 * (t1c1 / d1))
    S2mu = S2mu - (meta_d1 * (t1c1 / d1))

    t1c1 = 0

    # set up MLE analysis
    # get type 2 response counts
    # S1 responses
    nC_rS1, nI_rS1 = [], []
    for i in range(nRatings):
        nC_rS1.append(nR_S1[i])
        nI_rS1.append(nR_S2[i])

    # S2 responses
    nC_rS2 = [nR_S2[i + nRatings] for i in np.arange(nRatings)]
    nI_rS2 = [nR_S1[i + nRatings] for i in np.arange(nRatings)]

    # get type 2 probabilities
    C_area_rS1 = norm_cdf(t1c1, S1mu, S1sd)
    I_area_rS1 = norm_cdf(t1c1, S2mu, S2sd)

    C_area_rS2 = 1 - norm_cdf(t1c1, S2mu, S2sd)
    I_area_rS2 = 1 - norm_cdf(t1c1, S1mu, S1sd)

    t2c1x = [-np.inf]
    t2c1x.extend(t2c1[0 : (nRatings - 1)])
    t2c1x.append(t1c1)
    t2c1x.extend(t2c1[(nRatings - 1) :])
    t2c1x.append(np.inf)

    prC_rS1 = [
        (norm_cdf(t2c1x[i + 1], S1mu, S1sd) - norm_cdf(t2c1x[i], S1mu, S1sd))
        / C_area_rS1
        for i in range(nRatings)
    ]
    prI_rS1 = [
        (norm_cdf(t2c1x[i + 1], S2mu, S2sd) - norm_cdf(t2c1x[i], S2mu, S2sd))
        / I_area_rS1
        for i in range(nRatings)
    ]

    prC_rS2 = [
        (
            (1 - norm_cdf(t2c1x[nRatings + i], S2mu, S2sd))
            - (1 - norm_cdf(t2c1x[nRatings + i + 1], S2mu, S2sd))
        )
        / C_area_rS2
        for i in range(nRatings)
    ]
    prI_rS2 = [
        (
            (1 - norm_cdf(t2c1x[nRatings + i], S1mu, S1sd))
            - (1 - norm_cdf(t2c1x[nRatings + i + 1], S1mu, S1sd))
        )
        / I_area_rS2
        for i in range(nRatings)
    ]

    # calculate logL
    logL = 0.0
    for i in range(nRatings):
        logL = (
            logL
            + nC_rS1[i] * np.log(prC_rS1[i])
            + nI_rS1[i] * np.log(prI_rS1[i])
            + nC_rS2[i] * np.log(prC_rS2[i])
            + nI_rS2[i] * np.log(prI_rS2[i])
        )

    # returning -inf may cause optimize.minimize() to fail
    if np.isinf(logL) or np.isnan(logL):
        logL = -1e300

    return -logL


@overload
def metad(
    data: None,
    nRatings: int,
    nR_S1: None,
    nR_S2: None,
    stimuli: Union[list, np.ndarray],
    accuracy: Union[list, np.ndarray],
    confidence: Union[list, np.ndarray],
    within=None,
    between=None,
    subject=None,
    padAmount: Optional[float] = None,
    s: int = 1,
    padding: bool = True,
    collapse: Optional[int] = None,
    fncdf: Callable[..., float] = norm.cdf,
    fninv: Callable[..., float] = norm.ppf,
    verbose: int = 1,
) -> Union[dict, pd.DataFrame]:
    ...


@overload
def metad(
    data: None,
    nRatings: int,
    nR_S1: Union[list, np.ndarray],
    nR_S2: Union[list, np.ndarray],
    stimuli: None,
    accuracy: None,
    confidence: None,
    within=None,
    between=None,
    subject=None,
    padAmount: Optional[float] = None,
    s: int = 1,
    padding: bool = True,
    collapse: Optional[int] = None,
    fncdf: Callable[..., float] = norm.cdf,
    fninv: Callable[..., float] = norm.ppf,
    verbose: int = 1,
) -> pd.DataFrame:
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
    within: Optional[str] = None,
    between: Optional[str] = None,
    subject: Optional[str] = None,
    padAmount: Optional[float] = None,
    s: int = 1,
    padding: bool = True,
    collapse: Optional[int] = None,
    fncdf: Callable[..., float] = norm.cdf,
    fninv: Callable[..., float] = norm.ppf,
    verbose: int = 1,
) -> pd.DataFrame:
    ...


@pf.register_dataframe_method
def metad(
    data=None,
    nRatings=4,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
    within=None,
    between=None,
    subject=None,
    padAmount=None,
    nR_S1=None,
    nR_S2=None,
    s=1,
    padding=True,
    collapse=None,
    fncdf=norm_cdf,
    fninv=norm.ppf,
    verbose=0,
):
    """Estimate meta-d' using maximum likelihood estimation (MLE).

    This function is adapted from the transcription of fit_meta_d_MLE.m
    (Maniscalco & Lau, 2012) by Alan Lee: http://www.columbia.edu/~bsm2105/type2sdt/.

    Parameters
    ----------
    data :
        Dataframe. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    nRatings :
        Number of discrete ratings. If a continuous rating scale was used, and
        the number of unique ratings does not match `nRatings`, will convert to
        discrete ratings using :py:func:`metadpy.utils.discreteRatings`.
        Default is set to 4.
    stimuli :
        Name of the column containing the stimuli.
    accuracy :
        Name of the columns containing the accuracy.
    confidence :
        Name of the column containing the confidence ratings.
    within :
        Name of column containing the within factor (condition comparison).
    between :
        Name of column containing the between subject factor (group
        comparison).
    subject :
        Name of column containing the subject identifier (only required if a
        within-subject or a between-subject factor is provided).
    nR_S1, nR_S2 :
        These are vectors containing the total number of responses in
        each response category, conditional on presentation of S1 and S2. If
        nR_S1 = [100, 50, 20, 10, 5, 1], then when stimulus S1 was presented, the
        subject had the following response counts:
        * responded `'S1'`, rating=`3` : 100 times
        * responded `'S1'`, rating=`2` : 50 times
        * responded `'S1'`, rating=`1` : 20 times
        * responded `'S2'`, rating=`1` : 10 times
        * responded `'S2'`, rating=`2` : 5 times
        * responded `'S2'`, rating=`3` : 1 time

        The ordering of response / rating counts for S2 should be the same as
        it is for S1. e.g. if nR_S2 = [3, 7, 8, 12, 27, 89], then when stimulus S2
        was presented, the subject had the following response counts:
        * responded `'S1'`, rating=`3` : 3 times
        * responded `'S1'`, rating=`2` : 7 times
        * responded `'S1'`, rating=`1` : 8 times
        * responded `'S2'`, rating=`1` : 12 times
        * responded `'S2'`, rating=`2` : 27 times
        * responded `'S2'`, rating=`3` : 89 times

    s :
        Ratio of standard deviations for type 1 distributions as:
        `s = np.std(S1) / np.std(S2)`. If not specified, s is set to a default
        value of 1. For most purposes, it is recommended to set `s=1`. See
        http://www.columbia.edu/~bsm2105/type2sdt for further discussion.
    padding :
        If `True`, a small value will be added to the counts to avoid problems
        during fit.
    padAmount :
        The value to add to each response count if padding is set to 1.
        Default value is 1/(2*nRatings)
    collapse :
        If an integer `N` is provided, will collpase ratings to avoid zeros by
        summing every `N` consecutive ratings. Default set to `None`.
    fncdf :
        A function handle for the CDF of the type 1 distribution. If not
        specified, fncdf defaults to :py:func:`scipy.stats.norm.cdf()`.
    fninv :
        A function handle for the inverse CDF of the type 1 distribution. If
        not specified, fninv defaults to :py:func:`scipy.stats.norm.ppf()`.
    verbose :
        Level of algorithm's verbosity (can be `0`, `1`, `2` or `3`):
        * 0 (default) : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations.
        * 3 : display progress during iterations (more complete report).

    Returns
    -------
    results :
        In the following, S1 and S2 represent the distributions of evidence
        generated by stimulus classes S1 and S2:
        * `'dprime'` : `mean(S2) - mean(S1)`, in
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
    if verbose:
        print("Estimate meta-d' using maximum likelihood estimation (MLE).")
    if isinstance(data, pd.DataFrame):
        if padAmount is None:
            padAmount = 1 / (2 * nRatings)

        if subject is None:
            subject = "Subject"
            data.loc[:, subject] = "Subject 1"
        if within is None:
            within = "Within"
            data.loc[:, within] = "Condition 1"
        if between is None:
            between = "Between"
            data.loc[:, between] = "Group 1"
    else:
        if (nR_S1 is not None) & (nR_S2 is not None):
            data = pd.DataFrame(
                {
                    "nR_S1": [nR_S1],
                    "nR_S2": [nR_S2],
                }
            )
        elif (stimuli is not None) & (accuracy is not None) & (confidence is not None):
            data = pd.DataFrame(
                {"Stimuli": stimuli, "Accuracy": accuracy, "Confidence": confidence}
            )
            stimuli, accuracy, confidence = "Stimuli", "Accuracy", "Confidence"
        subject, within, between = "Subject", "Within", "Between"
        data[subject] = "Subject 1"
        data[within] = "Condition 1"
        data[between] = "Group 1"

    if verbose:
        print(f"- n Subjects: {data[subject].nunique()}")
        print(f"- n Conditions: {data[within].nunique()}")
        print(f"- n Groups: {data[between].nunique()}")
    if (padding is True) & verbose:
        tqdm.write("... Using padding to avoid fitting errors.")
    if (collapse is True) & verbose:
        tqdm.write("... Collapse across rating categories.")

    results_df = pd.DataFrame([])

    if verbose:
        pbar = tqdm(data[subject].unique(), position=0, leave=True, file=sys.stdout)
    else:
        pbar = data[subject].unique()

    for sub in pbar:

        if verbose:
            pbar.set_description(f"Fitting model on subject: {sub}")

        for cond in data[within].unique():

            for group in data[between].unique():

                if (nR_S1 is None) & (nR_S2 is None):
                    this_df = data[
                        (data[between] == group)
                        & (data[within] == cond)
                        & (data[subject] == sub)
                    ]

                    nR_S1, nR_S2 = trials2counts(
                        data=this_df,
                        stimuli=stimuli,
                        accuracy=accuracy,
                        confidence=confidence,
                        nRatings=nRatings,
                        padding=padding,
                        padAmount=padAmount,
                    )

                if (len(nR_S1) % 2) != 0:
                    raise ValueError(
                        "input arrays must have an even number of elements"
                    )
                if len(nR_S1) != len(nR_S2):
                    raise ValueError(
                        "input arrays must have the same number of elements"
                    )

                if (padding is False) & (collapse is None):
                    if any(np.array(nR_S1) == 0) or any(np.array(nR_S2) == 0):
                        import warnings

                        warnings.warn(
                            (
                                "Your inputs contain zeros and is not corrected."
                                "This may interfere with proper estimation of meta-d."
                                "See docstrings for more information."
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

                if nRatings is None:
                    nRatings = int(
                        len(nR_S1) / 2
                    )  # number of ratings in the experiment
                nCriteria = int(2 * nRatings - 1)  # number criteria to be fitted

                try:
                    results_dict = fit_metad(
                        nR_S1=nR_S1,
                        nR_S2=nR_S2,
                        nRatings=nRatings,
                        nCriteria=nCriteria,
                        s=s,
                        verbose=verbose,
                        fninv=fninv,
                        fncdf=fncdf,
                    )
                    # Filter dictionary and convert into pd.DataFrame
                    results = pd.DataFrame(
                        {
                            k: [results_dict[k]]
                            for k in ["dprime", "meta_d", "m_ratio", "m_diff"]
                        }
                    )
                except ValueError:
                    import warnings

                    warnings.warn(
                        (
                            f"Error when fitting data for subject {str(sub)} - "
                            f"Condition {str(cond)} - Group {str(group)}."
                            "Returning NaNs as a result"
                        )
                    )
                    results = pd.DataFrame(
                        {
                            "dprime": [None],
                            "meta_d": [None],
                            "m_ratio": [None],
                            "m_diff": [None],
                        }
                    )

                if data[subject].nunique() > 1:
                    results[subject] = sub
                if data[within].nunique() > 1:
                    results[within] = cond
                if data[between].nunique() > 1:
                    results[between] = group

                results_df = pd.concat([results_df, results])

                nR_S1, nR_S2 = None, None

    return results_df


def fit_metad(
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
    nRatings: int,
    nCriteria: Optional[int] = None,
    s: int = 1,
    verbose: int = 0,
    fninv: Callable = norm.ppf,
    fncdf: Callable = norm_cdf,
) -> Dict:
    """Fit metad model using MLE.

    Parameters
    ----------
    nR_S1, nR_S2 :
        These are vectors containing the total number of responses in
        each response category, conditional on presentation of S1 and S2. If
        nR_S1 = [100, 50, 20, 10, 5, 1], then when stimulus S1 was presented, the
        subject had the following response counts:
        * responded `'S1'`, rating=`3` : 100 times
        * responded `'S1'`, rating=`2` : 50 times
        * responded `'S1'`, rating=`1` : 20 times
        * responded `'S2'`, rating=`1` : 10 times
        * responded `'S2'`, rating=`2` : 5 times
        * responded `'S2'`, rating=`3` : 1 time

        The ordering of response / rating counts for S2 should be the same as
        it is for S1. e.g. if nR_S2 = [3, 7, 8, 12, 27, 89], then when stimulus S2
        was presented, the subject had the following response counts:
        * responded `'S1'`, rating=`3` : 3 times
        * responded `'S1'`, rating=`2` : 7 times
        * responded `'S1'`, rating=`1` : 8 times
        * responded `'S2'`, rating=`1` : 12 times
        * responded `'S2'`, rating=`2` : 27 times
        * responded `'S2'`, rating=`3` : 89 times
    nRatings :
        Number of discrete ratings. If a continuous rating scale was used, and
        the number of unique ratings does not match `nRatings`, will convert to
        discrete ratings using :py:func:`metadpy.utils.discreteRatings`.
        Default is set to 4.
    nCriteria :
        (Optional) Number criteria to be fitted. If `None`, the number of criteria is
        set to `nCriteria = int(2 * nRatings - 1)`.
    s :
        Ratio of standard deviations for type 1 distributions as:
        `s = np.std(S1) / np.std(S2)`. If not specified, s is set to a default
        value of 1. For most purposes, it is recommended to set `s=1`. See
        http://www.columbia.edu/~bsm2105/type2sdt for further discussion.
    verbose :
        Level of algorithm's verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.
            * 3 : display progress during iterations (more complete report).
    fninv :
        A function handle for the inverse CDF of the type 1 distribution. If
        not specified, fninv defaults to :py:func:`scipy.stats.norm.ppf()`.
    fncdf :
        A function handle for the CDF of the type 1 distribution. If not
        specified, fncdf defaults to :py:func:`scipy.stats.norm.cdf()`.

    Returns
    -------
    results :
        In the following, S1 and S2 represent the distributions of evidence generated
        by stimulus classes S1 and S2:
        * `'dprime'` : `mean(S2) - mean(S1)`, in root-mean-square(sd(S1), sd(S2))
            units
        * `'s'` : `sd(S1) / sd(S2)`
        * `'meta_d'` : meta-d' in RMS units
        * `'m_diff'` : `meta_da - da`
        * `'m_ratio'` : `meta_da / da`
        * `'meta_ca'` : type 1 criterion for meta-d' fit, RMS units.
        * `'t2ca_rS1'` : type 2 criteria of "S1" responses for meta-d' fit, RMS
        units.
        * `'t2ca_rS2'` : type 2 criteria of "S2" responses for meta-d' fit,
            RMS units.
        * `'logL'` : log likelihood of the data fit
        * `'est_HR2_rS1'` : estimated (from meta-d' fit) type 2 hit rates for S1
        responses.
        * `'obs_HR2_rS1'` : actual type 2 hit rates for S1 responses.
        * `'est_FAR2_rS1'` : estimated type 2 false alarm rates for S1 responses.
        * `'obs_FAR2_rS1'` : actual type 2 false alarm rates for S1 responses.
        * `'est_HR2_rS2'` : estimated type 2 hit rates for S2 responses.
        * `'obs_HR2_rS2'` : actual type 2 hit rates for S2 responses.
        * `'est_FAR2_rS2'` : estimated type 2 false alarm rates for S2 responses.
        * `'obs_FAR2_rS2'` : actual type 2 false alarm rates for S2 responses.

    """
    if nCriteria is None:
        nCriteria = int(2 * nRatings - 1)

    # parameters
    # meta-d' - 1
    # t2c - nCriteria-1
    # constrain type 2 criteria values, such that t2c(i) is always <= t2c(i+1)
    # -->  t2c(i+1) >= t2c(i) + 1e-5 (i.e. very small deviation from equality)
    A, ub, lb = [], [], []
    for ii in range(nCriteria - 2):
        tempArow: List[int] = []
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
    guess.extend(list(t2c1 - (meta_d1 * (t1c1 / d1))))

    # other inputs for the minimization function
    bounds = Bounds(LB, UB)
    linear_constraint = LinearConstraint(A, lb, ub)

    # minimization of negative log-likelihood
    results = minimize(
        fit_meta_d_logL,
        guess,
        args=(nR_S1, nR_S2, nRatings, d1, t1c1, s),
        method="trust-constr",
        jac="2-point",
        hess=SR1(),
        constraints=[linear_constraint],
        options={"verbose": verbose},
        bounds=bounds,
    )

    # quickly process some of the output
    meta_d1 = results.x[0]
    t2c1 = results.x[1:] + (meta_d1 * (t1c1 / d1))
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

    mt1c1 = meta_d1 * (t1c1 / d1)

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
    fit["dprime"] = np.sqrt(2 / (1 + s**2)) * s * d1
    fit["s"] = s
    fit["meta_d"] = np.sqrt(2 / (1 + s**2)) * s * meta_d1
    fit["m_diff"] = fit["meta_d"] - fit["dprime"]
    fit["m_ratio"] = fit["meta_d"] / fit["dprime"]

    fit["meta_ca"] = (np.sqrt(2) * s / np.sqrt(1 + s**2)) * mt1c1

    t2ca = (np.sqrt(2) * s / np.sqrt(1 + s**2)) * np.array(t2c1)
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

    return fit
