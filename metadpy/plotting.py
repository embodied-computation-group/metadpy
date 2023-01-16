# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import norm


def plot_confidence(
    nR_S1: Union[list, np.ndarray],
    nR_S2: Union[list, np.ndarray],
    fitModel: Optional[dict] = None,
    ax: Axes = None,
) -> Axes:
    """Plot nR_S1 and nR_S2 confidence ratings.

    Parameters
    ----------
    nR_S1 : 1d array-like
        Confience ratings (stimuli 1).
    nR_S2 : 1d array-like
        Confidence ratings (stimuli 2).
    fitModel : dict or None
        Dictionary returned by :py:funct:`metadpy.mle.fit_metad()`. If
        provided, the estimated ratings will be plotted toghether with the
        observed data.
    ax : `Matplotlib.Axes` or None
        Where to draw the plot. Default is `None` (create a new figure).

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.

    """
    if fitModel is not None:
        if isinstance(fitModel, dict):
            # Retrieve estimate of nRatings for correct and incorrect
            mu1 = fitModel["meta_d"] / 2
            mean_c2 = (fitModel["t2ca_rS2"] + np.abs(np.flip(fitModel["t2ca_rS1"]))) / 2
            I_area = 1 - norm.cdf(0, -mu1, 1)
            C_area = 1 - norm.cdf(0, mu1, 1)
            allC = np.hstack([0, mean_c2, np.inf])
            I_prop_model = (
                norm.cdf(allC[1:], -mu1, 1) - norm.cdf(allC[:-1], -mu1, 1)
            ) / I_area
            C_prop_model = (
                norm.cdf(allC[1:], mu1, 1) - norm.cdf(allC[:-1], mu1, 1)
            ) / C_area
        else:
            raise ValueError(
                "You should provided a dictionary. " "See metadpy.sdt.metad() for help."
            )

    if len(nR_S1) != len(nR_S2):
        raise ValueError("nR_S1 and nR_S2 should have same length")

    # Get the number of ratings
    nRratings = int(len(nR_S1) / 2)

    # Collapse across two stimuli for correct and incorrect responses
    # this gives flipped corrects followed by incorrects
    obsCount = nR_S1 + np.flip(nR_S2)
    C_prop_data = np.flip(obsCount[:nRratings]) / sum(obsCount[:nRratings])
    I_prop_data = obsCount[nRratings:] / sum(obsCount[nRratings:])

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(
        x=np.arange(0.8, nRratings),
        height=C_prop_data,
        color="#5f9e6e",
        width=0.4,
        ec="k",
        label="Obs Correct",
    )
    ax.bar(
        x=np.arange(1.2, nRratings + 0.5),
        height=I_prop_data,
        color="#b55d60",
        width=0.4,
        ec="k",
        label="Obs Incorrect",
    )
    if fitModel is not None:
        ax.plot(
            np.arange(1.2, nRratings + 0.5),
            I_prop_model,
            "o",
            color="w",
            markeredgecolor="#b55d60",
            markersize=16,
            markeredgewidth=3,
            label="Est Incorrect",
        )
        ax.plot(
            np.arange(0.8, nRratings),
            C_prop_model,
            "o",
            color="w",
            markeredgecolor="#5f9e6e",
            markersize=16,
            markeredgewidth=3,
            label="Est Correct",
        )
    ax.set_ylabel("P$_{(Confidence=y|Outcome)}$")
    ax.set_xlabel("Confidence level")
    ax.set_title("Confidence ratings\n and task performances")
    ax.set_xticks(range(1, nRratings + 1))

    return ax


def plot_roc(
    nR_S1: Union[list, np.ndarray],
    nR_S2: Union[list, np.ndarray],
    fitModel: Optional[dict] = None,
    ax: Axes = None,
) -> Axes:
    """Function to plot type2 ROC curve from observed an estimated data fit.

    Parameters
    ----------
    nR_S1 : 1d array-like
        Number of ratings for signal 1 (correct and incorrect).
    nR_S2 : 1d array-like
        Number of ratings for signal 2 (correct and incorrect).
    fitModel : dict or None
        Dictionary returned by :py:func:`metadpy.mle.fit_metad()`. If
        provided, the estimated ratings will be plotted toghether with the
        observed data.
    ax : `Matplotlib.Axes` or None
        Where to draw the plot. Default is `None` (create a new figure).

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes containing the plot.

    """
    if fitModel is None:

        nRatings = int(len(nR_S1) / 2)

        # Find incorrect observed ratings
        I_nR_rS2 = nR_S1[nRatings:]
        I_nR_rS1 = np.flip(nR_S2[:nRatings])
        I_nR = I_nR_rS2 + I_nR_rS1

        # Find correct observed ratings
        C_nR_rS2 = nR_S2[nRatings:]
        C_nR_rS1 = np.flip(nR_S1[:nRatings])
        C_nR = C_nR_rS2 + C_nR_rS1

        # Calculate type 2 hits and false alarms
        obs_FAR2_rS2, obs_HR2_rS2, obs_FAR2_rS1, obs_HR2_rS1, obs_FAR2, obs_HR2 = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(nRatings):
            obs_FAR2_rS2.append(sum(I_nR_rS2[i:]) / sum(I_nR_rS2))
            obs_HR2_rS2.append(sum(C_nR_rS2[i:]) / sum(C_nR_rS2))
            obs_FAR2_rS1.append(sum(I_nR_rS1[i:]) / sum(I_nR_rS1))
            obs_HR2_rS1.append(sum(C_nR_rS1[i:]) / sum(C_nR_rS1))
            obs_FAR2.append(sum(I_nR[i:]) / sum(I_nR))
            obs_HR2.append(sum(C_nR[i:]) / sum(C_nR))

        obs_FAR2.append(0)
        obs_HR2.append(0)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.fill_between(x=obs_FAR2, y1=obs_HR2, color="lightgray", alpha=0.5)
        ax.plot(
            obs_FAR2, obs_HR2, "ko-", linewidth=1.5, markersize=12, label="Observed"
        )
        ax.set_title("Type 2 ROC curve")
        ax.set_ylabel("Type 2 P(correct)")
        ax.set_xlabel("Type 2 P(incorrect)")

    else:
        if not isinstance(fitModel, dict):
            raise ValueError(
                "You should provided a dictionary. " "See metadpy.mle.metad() for help."
            )
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Stimulus 1
        ax[0].plot([0, 1], [0, 1], "--", color="gray")
        ax[0].plot(
            fitModel["obs_FAR2_rS1"],
            fitModel["obs_HR2_rS1"],
            "ko-",
            linewidth=1.5,
            markersize=12,
            label="Observed",
        )
        ax[0].plot(
            fitModel["est_FAR2_rS1"],
            fitModel["est_HR2_rS1"],
            "bo-",
            linewidth=1.5,
            markersize=6,
            label="Estimated",
        )
        ax[0].set_title("Stimulus 1")
        ax[0].set_ylabel("Type 2 Hit Rate")
        ax[0].set_xlabel("Type 2 False Alarm Rate")
        ax[0].legend()

        # Stimulus 2
        ax[1].plot([0, 1], [0, 1], "--", color="gray")
        ax[1].plot(
            fitModel["obs_FAR2_rS2"],
            fitModel["obs_HR2_rS2"],
            "ko-",
            linewidth=1.5,
            markersize=12,
            label="Observed",
        )
        ax[1].plot(
            fitModel["est_FAR2_rS2"],
            fitModel["est_HR2_rS2"],
            "bo-",
            linewidth=1.5,
            markersize=6,
            label="Estimated",
        )
        ax[1].set_title("Stimulus 2")
        ax[1].set_ylabel("Type 2 Hit Rate")
        ax[1].set_xlabel("Type 2 False Alarm Rate")
        ax[1].legend()

    return ax
