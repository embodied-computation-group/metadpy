# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Any, List, Tuple, Union, overload

import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.stats import norm

from metadPy.utils import trials2counts


@overload
def scores(
    data: None,
    stimuli: Union[List[int], np.ndarray],
    responses: Union[List[int], np.ndarray],
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
    hits = sum(data[stimuli].astype("bool") & data[responses].astype("bool"))
    misses = sum(data[stimuli].astype("bool") & ~data[responses].astype("bool"))
    fas = sum(~data[stimuli].astype("bool") & data[responses].astype("bool"))
    crs = sum(~data[stimuli].astype("bool") & ~data[responses].astype("bool"))

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
    """Compute hit and false alarm rates.

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

    .. math:: fals alarms rate =
        \\frac{\\fals alarms}{s_{fals alarms + correct rejections}}

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


@overload
def roc_auc(
    data: None,
    stimuli: None,
    responses: None,
    accuracy: None,
    confidence: None,
    nRatings: None,
    nR_S1: Union[list, np.ndarray],
    nR_S2: Union[list, np.ndarray],
) -> float:
    ...


@overload
def roc_auc(
    data: None,
    stimuli: Union[list, np.ndarray],
    responses: Union[list, np.ndarray],
    accuracy: Union[list, np.ndarray],
    confidence: Union[list, np.ndarray],
    nRatings: None,
    nR_S1: None,
    nR_S2: None,
) -> float:
    ...


@overload
def roc_auc(
    data: pd.DataFrame,
    stimuli: str,
    responses: str,
    accuracy: str,
    confidence: str,
    nRatings: int,
    nR_S1: None,
    nR_S2: None,
) -> float:
    ...


@pf.register_dataframe_method
def roc_auc(
    data=None,
    stimuli=None,
    responses=None,
    accuracy=None,
    confidence=None,
    nRatings=None,
    nR_S1=None,
    nR_S2=None,
):
    """Calculate the area under the type 2 ROC curve given nR_S1 and nR_S2
    ratings counts.

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
    nRatings : int
        Total of available subjective ratings available for the subject. e.g.
        if subject can rate confidence on a scale of 1-4, then nRatings = 4.
        Default is `4`.
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
    if isinstance(data, pd.DataFrame):
        if stimuli is None:
            stimuli = "Stimuli"
        if responses is None:
            responses = "Responses"
        if confidence is None:
            confidence = "Confidence"

        nR_S1, nR_S2 = trials2counts(
            data=data,
            stimuli=stimuli,
            accuracy=accuracy,
            confidence=confidence,
            nRatings=nRatings,
        )

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
