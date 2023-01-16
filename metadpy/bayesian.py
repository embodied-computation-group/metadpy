# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
import sys
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union, overload

import arviz as az
import numpy as np
import pandas as pd
import pandas_flavor as pf
from arviz import InferenceData

from metadpy.sdt import criterion, dprime
from metadpy.utils import discreteRatings, trials2counts

if TYPE_CHECKING is True:
    from pymc.backends.base import MultiTrace
    from pymc.model import Model


@overload
def hmetad(
    data: None,
    nR_S1: Union[List, np.ndarray],
    nR_S2: Union[List, np.ndarray],
    nRatings: Optional[int],
    subject: None,
    within: None,
    nbins: int,
    padding: bool = False,
    padAmount: Optional[float] = None,
    output: str = "model",
) -> "Tuple[Union[Model, Callable], Optional[Union[InferenceData, MultiTrace]]]":
    ...


@overload
def hmetad(
    data: pd.DataFrame,
    stimuli: str,
    accuracy: str,
    confidence: str,
    nRatings: Optional[int],
    subject: None,
    within: None,
    nbins: int,
    padding: bool = False,
    padAmount: Optional[float] = None,
    output: str = "model",
) -> "Tuple[Union[Model, Callable], Optional[Union[InferenceData, MultiTrace]]]":
    ...


@overload
def hmetad(
    data: pd.DataFrame,
    stimuli: str,
    accuracy: str,
    confidence: str,
    nRatings: Optional[int],
    subject: str,
    within: None,
    nbins: int,
    padding: bool = False,
    padAmount: Optional[float] = None,
    output: str = "model",
) -> "Tuple[Union[Model, Callable], Optional[Union[InferenceData, MultiTrace]]]":
    ...


@overload
def hmetad(
    data: pd.DataFrame,
    stimuli: str,
    accuracy: str,
    confidence: str,
    nRatings: Optional[int],
    subject: str,
    within: str,
    nbins: int,
    padding: bool = False,
    padAmount: Optional[float] = None,
    sample_model: bool = True,
    output: str = "model",
) -> "Tuple[Union[Model, Callable], Optional[Union[InferenceData, MultiTrace]]]":
    ...


@pf.register_dataframe_method
def hmetad(
    data=None,
    nR_S1=None,
    nR_S2=None,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
    nRatings=None,
    within=None,
    between=None,
    subject=None,
    nbins=4,
    padding=False,
    padAmount=None,
    sample_model=True,
    output: str = "model",
    num_samples: int = 1000,
    num_chains: int = 4,
    **kwargs
):
    """Estimate parameters of the Bayesian meta-d' model with hyperparametes at the
    group level.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame` | None
        Dataframe. Note that this function can also directly be used as a Pandas
        method, in which case this argument is no longer needed.
    nR_S1 : 1d array-like, list, string | None
        Confience ratings (stimuli 1, correct and incorrect).
    nR_S2 : 1d array-like, list, string | None
        Confience ratings (stimuli 2, correct and incorrect).
    stimuli : string | None
        Name of the column containing the stimuli.
    accuracy : string | None
        Name of the columns containing the accuracy.
    confidence : string | None
        Name of the column containing the confidence ratings.
    nRatings : int | None
        Number of discrete ratings. If a continuous rating scale was used, and
        the number of unique ratings does not match `nRatings`, will convert to
        discrete ratings using :py:func:`metadpy.utils.discreteRatings`.
    within : string | None
        Name of column containing the within factor (condition comparison).
    between : string | None
        Name of column containing the between subject factor (group
        comparison).
    subject : string | None
        Name of column containing the subject identifier (only required if a
        within-subject or a between-subject factor is provided).
    nbins : int
        If a continuous rating scale was using, `nbins` define the number of
        discrete ratings when converting using
        :py:func:`metadpy.utils.discreteRatings`. The default value is `4`.
    padding : boolean
        If `True`, each response count in the output has the value of padAmount
        added to it. Padding cells is desirable if trial counts of 0 interfere
        with model fitting. If False, trial counts are not manipulated and 0s
        may be present in the response count output. Default value for padding
        is `False`.
    padAmount : float or None
        The value to add to each response count if padCells is set to 1.
        Default value is 1/(2*nRatings)
    sample_model : boolean
        If `False`, only the model is returned without sampling.
    output : str
        The kind of outpute expected. If `"model"`, will return the model function and
        the traces. If `"dataframe"`, will return a dataframe containing `d` (dprime),
        `c` (criterion), `meta_d` (the meta-d prime) and `m_ratio` (`meta_d/d`).
    num_samples : int
        The number of samples per chains to draw (defaults to `1000`).
    num_chains : int
        The number of chains (defaults to `4`).
    **kwargs : keyword arguments
        All keyword arguments are passed to `func::pymc.sampling.sample`.

    Returns
    -------
    If `output="model"`:

    model : :py:class:`pymc.Model` instance
        The model PyMC as a :py:class:`pymc.Model`.
    traces : :py:class:`pymc.backends.base.MultiTrace` |
        :py:class:`arviz.InferenceData` | None
        A `MultiTrace` or `ArviZ InferenceData` object that contains the samples. Only
        returned if `sample_model` is set to `True`, otherwise set to None.

    or

    results : pd.DataFrame
        If `output="dataframe"`, :py:class:`pandas.DataFrame` containing the values for
        the following variables:

        * d-prime (d)
        * criterion (c)
        * meta-d' (meta_d)
        * m-ratio (m_ratio)

    Examples
    --------
    1. Subject-level

    Notes
    -----
    This function will compute hierarchical Bayesian estimation of metacognitive
    efficiency as described in [1]_. The model can be fitter at the subject level, at
    the group level and can account for repeated measures by providing the corresponding
    `subject`, `between` and `within` factors.

    If the confidence levels have more unique values than `nRatings`, the confience
    column will be discretized using py:func:`metadpy.utils.discreteRatings`.

    Raises
    ------
    ValueError
        When the number of ratings is not provided.
        If data is None and nR_S1 or nR_S2 not provided.
        If the backend is not `"numpyro"` or `"pymc"`.

    References
    ----------
    .. [1] Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation of
      metacognitive efficiency from confidence ratings, Neuroscience of
      Consciousness, 3(1) nix007, https://doi.org/10.1093/nc/nix007.

    """
    modelScript = os.path.dirname(__file__) + "/models/"
    sys.path.append(modelScript)

    if (nR_S1 is not None) & (nR_S2 is not None):
        nR_S1, nR_S2 = np.asarray(nR_S1), np.asarray(nR_S2)
        if nRatings is not None:
            assert len(nR_S1) / 2 == nRatings
        else:
            nRatings = len(nR_S1) / 2

    if nRatings is None:
        raise ValueError("You should provide the number of ratings")

    if data is None:
        if (nR_S1 is None) or (nR_S2 is None):
            raise ValueError(
                "If data is None, you should provide"
                " the nR_S1 and nR_S2 vectors instead."
            )
    else:
        if data[confidence].nunique() > nRatings:

            # If a continuous rating scale was used (if N unique ratings > nRatings)
            # transform confidence to discrete ratings
            print(
                (
                    "The confidence columns contains more unique values than nRatings. "
                    "The ratings are going to be discretized using "
                    "metadpy.utils.discreteRatings()"
                )
            )
            new_ratings, _ = discreteRatings(data[confidence].to_numpy(), nbins=nbins)
            data.loc[:, confidence] = new_ratings

    ###############
    # Subject level
    if (within is None) & (between is None) & (subject is None):

        if data is not None:
            nR_S1, nR_S2 = trials2counts(
                data=data,
                stimuli=stimuli,
                accuracy=accuracy,
                confidence=confidence,
                nRatings=nRatings,
                padding=padding,
                padAmount=padAmount,
            )

        pymcData = extractParameters(np.asarray(nR_S1), np.asarray(nR_S2))

        from subject_level_pymc import hmetad_subjectLevel

        model_output = hmetad_subjectLevel(
            pymcData,
            sample_model=sample_model,
            num_chains=num_chains,
            num_samples=num_samples,
        )

    #############
    # Group level
    elif (within is None) & (between is None) & (subject is not None):

        # pymcData = preprocess_group(
        #     data, subject, stimuli, accuracy, confidence, nRatings
        # )

        raise ValueError(
            "Invalid backend provided - This model is not implemented yet."
        )

    ###################
    # Repeated-measures
    elif (within is not None) & (between is None) & (subject is not None):

        # pymcData = preprocess_rm1way(
        #     data, subject, within, stimuli, accuracy, confidence, nRatings
        # )

        raise ValueError("Invalid backend provided - This model is not implemented yet")

    ##########
    # Sampling
    if sample_model is True:
        model, traces = model_output

        if output == "model":
            return model, traces
        elif output == "dataframe":
            return pd.DataFrame(
                {
                    "d": [pymcData["d1"]],
                    "c": [pymcData["c1"]],
                    "meta_d": [
                        az.summary(traces, var_names=["meta_d"])["mean"]["meta_d"]
                    ],
                    "m_ratio": [
                        az.summary(traces, var_names=["meta_d"])["mean"]["meta_d"]
                        / pymcData["d1"]
                    ],
                }
            )
    else:
        return model_output, None


def extractParameters(
    nR_S1: Union[List[int], np.ndarray], nR_S2: Union[List[int], np.ndarray]
) -> Dict:
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
        Dictionary of rates and task parameters.

    See also
    --------
    hmetad
    """
    if isinstance(nR_S1, list):
        nR_S1 = np.array(nR_S1, dtype=float)
    if isinstance(nR_S2, list):
        nR_S2 = np.array(nR_S2, dtype=float)

    Tol = 1e-05
    nratings = int(len(nR_S1) / 2)

    # Adjust to ensure non-zero counts for type 1 d' point estimate
    adj_f = 1 / ((nratings) * 2)

    nR_S1_adj = nR_S1 + adj_f
    nR_S2_adj = nR_S2 + adj_f

    ratingHR: List[float] = []
    ratingFAR: List[float] = []
    for c in range(1, int(nratings * 2)):
        ratingHR.append(sum(nR_S2_adj[c:]) / sum(nR_S2_adj))
        ratingFAR.append(sum(nR_S1_adj[c:]) / sum(nR_S1_adj))

    d1 = dprime(
        data=None,
        stimuli=None,
        responses=None,
        hit_rate=ratingHR[nratings - 1],
        fa_rate=ratingFAR[nratings - 1],
    )
    c1 = criterion(
        data=None,
        hit_rate=ratingHR[nratings - 1],
        fa_rate=ratingFAR[nratings - 1],
        stimuli=None,
        responses=None,
    )
    counts = np.hstack([nR_S1, nR_S2])

    # Type 1 counts
    N = sum(counts[: (nratings * 2)])
    S = sum(counts[(nratings * 2) : (nratings * 4)])
    H = sum(counts[(nratings * 3) : (nratings * 4)])
    M = sum(counts[(nratings * 2) : (nratings * 3)])
    FA = sum(counts[(nratings) : (nratings * 2)])
    CR = sum(counts[:(nratings)])

    # Data preparation for model
    data = {
        "d1": d1,
        "c1": c1,
        "counts": counts,
        "nratings": nratings,
        "Tol": Tol,
        "FA": FA,
        "CR": CR,
        "M": M,
        "H": H,
        "N": N,
        "S": S,
    }

    return data


# TODO: when implementing group level fitting, the following wrapper will be usefull.
# def preprocess_group(
#     data: pd.DataFrame,
#     subject: str,
#     stimuli: str,
#     accuracy: str,
#     confidence: str,
#     nRatings: int,
# ) -> Dict:
#     """Preprocess group data.

#     Parameters
#     ----------
#     data : :py:class:`pandas.DataFrame` or None
#         Dataframe. Note that this function can also directly be used as a
#         Pandas method, in which case this argument is no longer needed.
#     subject : string or None
#         Name of column containing the subject identifier (only required if a
#         within-subject or a between-subject factor is provided).
#     stimuli : string or None
#         Name of the column containing the stimuli.
#     accuracy : string or None
#         Name of the columns containing the accuracy.
#     confidence : string or None
#         Name of the column containing the confidence ratings.
#     nRatings : int or None
#         Number of discrete ratings. If a continuous rating scale was used, and
#         the number of unique ratings does not match `nRatings`, will convert to
#         discrete ratings using :py:func:`metadpy.utils.discreteRatings`.

#     Return
#     ------
#     pymcData : Dict

#     """
#     pymcData = {
#         "d1": [],
#         "c1": [],
#         "nSubj": data[subject].nunique(),
#         "subID": np.arange(data[subject].nunique(), dtype="int"),
#         "hits": [],
#         "falsealarms": [],
#         "s": [],
#         "n": [],
#         "counts": [],
#         "nRatings": nRatings,
#         "Tol": 1e-05,
#         "cr": [],
#         "m": [],
#     }

#     for sub in data[subject].unique():
#         nR_S1, nR_S2 = trials2counts(
#             data=data[data[subject] == sub],
#             stimuli=stimuli,
#             accuracy=accuracy,
#             confidence=confidence,
#             nRatings=nRatings,
#         )

#         this_data = extractParameters(nR_S1, nR_S2)
#         pymcData["d1"].append(this_data["d1"])
#         pymcData["c1"].append(this_data["c1"])
#         pymcData["s"].append(this_data["S"])
#         pymcData["n"].append(this_data["N"])
#         pymcData["m"].append(this_data["M"])
#         pymcData["cr"].append(this_data["CR"])
#         pymcData["counts"].append(this_data["counts"])
#         pymcData["hits"].append(this_data["H"])
#         pymcData["falsealarms"].append(this_data["FA"])

#     pymcData["d1"] = np.array(pymcData["d1"], dtype="float")
#     pymcData["c1"] = np.array(pymcData["c1"], dtype="float")
#     pymcData["s"] = np.array(pymcData["s"], dtype="int")
#     pymcData["n"] = np.array(pymcData["n"], dtype="int")
#     pymcData["m"] = np.array(pymcData["m"], dtype="int")
#     pymcData["cr"] = np.array(pymcData["cr"], dtype="int")
#     pymcData["counts"] = np.array(pymcData["counts"], dtype="int")
#     pymcData["hits"] = np.array(pymcData["hits"], dtype="int")
#     pymcData["falsealarms"] = np.array(pymcData["falsealarms"], dtype="int")
#     pymcData["nSubj"] = data[subject].nunique()
#     pymcData["subID"] = np.arange(pymcData["nSubj"], dtype="int")

#     return pymcData


# def preprocess_rm1way(
#     data: pd.DataFrame,
#     subject: str,
#     stimuli: str,
#     within: str,
#     accuracy: str,
#     confidence: str,
#     nRatings: int,
# ) -> Dict:
#     """Preprocess repeated measures data.

#     Parameters
#     ----------
#     data : :py:class:`pandas.DataFrame`
#         Dataframe. Note that this function can also directly be used as a
#         Pandas method, in which case this argument is no longer needed.
#     subject : string
#         Name of column containing the subject identifier (only required if a
#         within-subject or a between-subject factor is provided).
#     stimuli : string
#         Name of the column containing the stimuli.
#     within : string
#         Name of column containing the within factor (condition comparison).
#     accuracy : string
#         Name of the columns containing the accuracy.
#     confidence : string
#         Name of the column containing the confidence ratings.
#     nRatings : int
#         Number of discrete ratings. If a continuous rating scale was used, and
#         the number of unique ratings does not match `nRatings`, will convert to
#         discrete ratings using :py:func:`metadpy.utils.discreteRatings`.

#     Return
#     ------
#     pymcData : Dict

#     """
#     pymcData = {
#         "nSubj": data[subject].nunique(),
#         "subID": [],
#         "nCond": data[within].nunique(),
#         "condition": [],
#         "hits": [],
#         "falsealarms": [],
#         "s": [],
#         "n": [],
#         "nRatings": nRatings,
#         "Tol": 1e-05,
#         "cr": [],
#         "m": [],
#     }
#     pymcData["counts"] = np.zeros(
#         (pymcData["nSubj"], pymcData["nCond"], pymcData["nRatings"] * 4)
#     )
#     pymcData["hits"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["falsealarms"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["s"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["n"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["m"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["cr"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["condition"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["subID"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["c1"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["d1"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))

#     for nSub, sub in enumerate(data[subject].unique()):
#         for nCond, cond in enumerate(data[within].unique()):
#             nR_S1, nR_S2 = trials2counts(
#                 data=data[(data[subject] == sub) & (data[within] == cond)],
#                 stimuli=stimuli,
#                 accuracy=accuracy,
#                 confidence=confidence,
#                 nRatings=nRatings,
#             )

#             this_data = extractParameters(nR_S1, nR_S2)
#             pymcData["subID"][nSub, nCond] = nSub
#             pymcData["condition"][nSub, nCond] = nCond
#             pymcData["s"][nSub, nCond] = this_data["S"]
#             pymcData["n"][nSub, nCond] = this_data["N"]
#             pymcData["m"][nSub, nCond] = this_data["M"]
#             pymcData["cr"][nSub, nCond] = this_data["CR"]
#             pymcData["hits"][nSub, nCond] = this_data["H"]
#             pymcData["falsealarms"][nSub, nCond] = this_data["FA"]
#             pymcData["c1"][nSub, nCond] = this_data["c1"]
#             pymcData["d1"][nSub, nCond] = this_data["d1"]
#             pymcData["counts"][nSub, nCond, :] = this_data["counts"]

#     pymcData["subID"] = np.array(pymcData["subID"], dtype="int")
#     pymcData["condition"] = np.array(pymcData["condition"], dtype="int")
#     pymcData["s"] = np.array(pymcData["s"], dtype="int")
#     pymcData["n"] = np.array(pymcData["n"], dtype="int")
#     pymcData["m"] = np.array(pymcData["m"], dtype="int")
#     pymcData["cr"] = np.array(pymcData["cr"], dtype="int")
#     pymcData["counts"] = np.array(pymcData["counts"], dtype="int")
#     pymcData["hits"] = np.array(pymcData["hits"], dtype="int")
#     pymcData["falsealarms"] = np.array(pymcData["falsealarms"], dtype="int")

#     return pymcData
