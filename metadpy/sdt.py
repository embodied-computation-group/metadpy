# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Any, List, Tuple, Union, overload

import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.stats import norm


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
    """Hits, misses, false alarms and correct rejection from stimuli and responses.

    Parameters
    ----------
    data :
        Dataframe containing one `stimuli` and one `response` column.
    stimuli :
        If a string is provided, should be the name of the column used as `stimuli`. If
        a list or an array is provided, should contain the boolean vectors for
        `stimuli`.
    responses :
        If a string is provided, should be the name of the column used as `responses`.
        If a list or an array is provided, should contain the boolean vector for
        `responses`.

    Returns
    -------
    hits, misses, fas, crs :
        The number of hits, misees, false alarms and correct rejections.

    Notes
    -----
    If a :py:class:`pandas.DataFrame` is provided, the function will search for a
    `stimuli`and a `responses` column by default if no other column names are provided.

    Raises
    ------
    ValueError:
        If no valid data is provided.

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
                    "If no data is provided, `stimuli` and `responses` should be"
                    " two boolean vectors with equal lengths."
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
    r"""Compute hit and false alarm rates.

    The values are automatically corrected to avoid d' infinity (see below).

    Parameters
    ----------
    data :
        Dataframe containing one `stimuli` and one `response` column.
    stimuli :
        If a string is provided, should be the name of the column used as
        `stimuli`. If a list or an array is provided, should contain the
        boolean vectors for `stimuli`. If `None` and `data` is a
        :py:class:`pandas.DataFrame`, will be set to `Stimuli` by default.
    responses :
        If a string is provided, should be the name of the column used as
        `responses`. If a list or an array is provided, should contain the
        boolean vector for `responses`. If `None` and `data` is a
        :py:class:`pandas.DataFrame`, will be set to `Responses` by default.
    hits :
        Hits.
    misses :
        Misses.
    fas :
        False alarms.
    crs :
        Correct rejections.
    correction :
        Avoid d' infinity by correcting false alarm and hit rate values if equal to 0
        or 1 using half inverse or 1 - half inverse.
        Half inverses values are defined by:
            half_hit = 0.5 / (hits + misses)
            half_fa = 0.5 / (fas + crs)
        Default is set to `True` (use correction).

    Returns
    -------
    hit_rate:
        Hit rate.
    fa_rate :
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

    See Also
    --------
    dprime, criterion, scores

    References
    ----------
    Adapted from: https://lindeloev.net/calculating-d-in-python-and-php/

    Raises
    ------
    ValueError:
        If no valid data are provided.

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
    data :
        Dataframe. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    hit_rate :
        Hit rate.
    fa_rate :
        False alarm rate.
    stimuli :
        Name of the column containing the stimuli. If `None` and `data` is a
        :py:class:`pandas.DataFrame`, will be set to `Stimuli` by default.
    responses :
        Name of the column containing the responses. If `None` and `data` is a
        :py:class:`pandas.DataFrame`, will be set to `Responses` by default.

    Returns
    -------
    dprime :
        The d' value.

    Notes
    -----
    The d' is a measure of the ability to discriminate a signal from noise.

    Raises
    ------
    ValueError
        If `stimuli` and `responses` are not srings when providing a data frame.
        If no data is provided.

    """
    if isinstance(data, pd.DataFrame):
        if stimuli is None:
            stimuli = "Stimuli"
        if responses is None:
            responses = "Responses"
        if (not isinstance(stimuli, str)) | (not isinstance(responses, str)):
            raise ValueError(
                (
                    "Parameters `stimuli` and `responses` must be strings"
                    "when `data` is a DataFrame."
                )
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
    data :
        Dataframe. Note that this function can also directly be used as a Pandas
        method, in which case this argument is no longer needed.
    hit_rate :
        Hit rate.
    fa_rate :
        False alarm rate.
    stimuli :
        Name of the column containing the stimuli.
    responses :
        Name of the column containing the responses.

    Returns
    -------
    dprime :
        The d' value.

    Raises
    ------
    ValueError
        If `data` is not a pd.DataFrame.

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
    """Calculate the area under the type 2 ROC curve given from confidence ratings.

    Parameters
    ----------
    data :
        Dataframe containing one `"stimuli"` and one `"response"` column. Different
        column names can also be provided using the `stimuli` and `responses`
        parameters.
    stimuli :
        If a string is provided, should be the name of the column used as `stimuli`. If
        a list or an array is provided, should contain the boolean vectors for
        `stimuli`. If `None` (default) and `data` is a :py:class:`pandas.DataFrame`,
        will be set to `Stimuli` by default.
    responses :
        If a string is provided, should be the name of the column used as `responses`.
        If a list or an array is provided, should contain the boolean vector for
        `responses`. If `None` (default) and `data` is a :py:class:`pandas.DataFrame`,
        will be set to `Responses` by default.
    accuracy :
        If a string is provided, should be the name of the column used as `accuracy`.
        If a list or an array is provided, should contain the boolean vector for
        `accuracy`. If `None` (default) and `data` is a :py:class:`pandas.DataFrame`,
        will be set to `Accuracy` by default. This parameter is optional if `stimuli`
        and `responses` are known.
    confidence :
        If a string is provided, should be the name of the column used as `confidence`.
        If a list or an array is provided, should contain the confidence ratings,
        matching the number of discret ratings provided in the `nRatings` parameter.
        If `None` (default) and `data` is a :py:class:`pandas.DataFrame`,
        will be set to `Confidence` by default.
    nRatings :
        Total of available subjective ratings available for the subject. e.g. if subject
        can rate confidence on a scale of 1-4, then nRatings = 4. Default is `None`.
    nR_S1 :
        Confience ratings (stimuli 1).
    nR_S2 :
        Confidence ratings (stimuli 2).

    Returns
    -------
    auc :
        Area under the type 2 ROC curve.

    Examples
    --------
    >>> nR_S1 = [36, 24, 17, 20, 10, 12, 9, 2]
    >>> nR_S2 = [1, 4, 10, 11, 19, 18, 28, 39]
    >>> roc_auc(nR_S1, nR_S2)
    0.6998064266356949

    Raises
    ------
    ValueError:
        If data and both nR_S1 and nR_S2 are missing.

    """
    if isinstance(data, pd.DataFrame):
        if stimuli is None:
            stimuli = "Stimuli"
        if responses is None:
            responses = "Responses"
        if confidence is None:
            confidence = "Confidence"

        # Compute accuracy
        accuracy = (data[stimuli] == data[responses]).to_numpy()
        conf = data[confidence].to_numpy()
        H2, FA2 = [], []
        for c in range(nRatings, 0, -1):
            H2.append((accuracy & (conf == c)).sum() + 1)
            FA2.append(((~accuracy) & (conf == c)).sum() + 1)

    else:
        if isinstance(nR_S1, list):
            nR_S1 = np.array(nR_S1)
        if isinstance(nR_S2, list):
            nR_S2 = np.array(nR_S2)
        if (nR_S1 is None) & (nR_S2 is None):
            raise ValueError("Either data or nR_s1 and nR_S2 should be provided.")

        if nRatings is None:
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
