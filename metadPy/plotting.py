# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import matplotlib.pyplot as plt


def plot_confidence(nR_S1, nR_S2, ax=None):
    """Plot nR_S1 and nR_S2 confidence ratings.

    Parameters
    ----------
    nR_S1 : 1d array-like
        Confience ratings (stimuli 1).
    nR_S2 : 1d array-like
        Confidence ratings (stimuli 2).
    ax : `Matplotlib.Axes` or None
        Where to draw the plot. Default is `None` (create a new figure).

    Returns
    -------
    ax : `Matplotlib.Axes` or None
        The figure.

    Examples
    --------
    """
    if len(nR_S1) != len(nR_S2):
        raise ValueError('nR_S1 and nR_S2 should have same length')

    # Get the number of ratings
    nRratings = int(len(nR_S1)/2)

    # Collapse across two stimuli for correct and incorrect responses
    # this gives flipped corrects followed by incorrects
    obsCount = nR_S1 + np.flip(nR_S2)
    C_prop_data = np.flip(obsCount[:nRratings])/sum(obsCount[:nRratings])
    I_prop_data = obsCount[nRratings:]/sum(obsCount[nRratings:])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(x=np.arange(.8, nRratings), height=C_prop_data, color='#5f9e6e',
           width=.4, ec="k", label='Correct')
    ax.bar(x=np.arange(1.2, nRratings+.5), height=I_prop_data, color='#b55d60',
           width=.4, ec="k", label='Incorrect')
    ax.set_ylabel('P$_{(Confidence=y|Outcome)}$')
    ax.set_xlabel('Confidence rating')
    ax.set_xticks(range(1, nRratings+1))

    return ax


def plot_roc(nR_S1, nR_S2, ax=None):
    """Function to plot type2 ROC curve from observed an estimated data fit.

    Parameters
    ----------
    nR_S1 : 1d array-like
        Number of ratings for signal 1 (correct and incorrect).
    nR_S2 : 1d array-like
        Number of ratings for signal 2 (correct and incorrect).

    Returns
    -------
    ax : `Matplotlib.Axes` or None
        The figure.

    Examples
    --------
    """
    nRatings = int(len(nR_S1)/2)

    # Find incorrect observed ratings
    I_nR_rS2 = nR_S1[nRatings:]
    I_nR_rS1 = np.flip(nR_S2[:nRatings])
    I_nR = I_nR_rS2 + I_nR_rS1

    # Find correct observed ratings
    C_nR_rS2 = nR_S2[nRatings:]
    C_nR_rS1 = np.flip(nR_S1[:nRatings])
    C_nR = C_nR_rS2 + C_nR_rS1

    # Calculate type 2 hits and false alarms
    obs_FAR2_rS2, obs_HR2_rS2, obs_FAR2_rS1, obs_HR2_rS1, obs_FAR2, obs_HR2 =\
        [], [], [], [], [], []
    for i in range(nRatings):
        obs_FAR2_rS2.append(sum(I_nR_rS2[i:])/sum(I_nR_rS2))
        obs_HR2_rS2.append(sum(C_nR_rS2[i:])/sum(C_nR_rS2))
        obs_FAR2_rS1.append(sum(I_nR_rS1[i:])/sum(I_nR_rS1))
        obs_HR2_rS1.append(sum(C_nR_rS1[i:])/sum(C_nR_rS1))
        obs_FAR2.append(sum(I_nR[i:])/sum(I_nR))
        obs_HR2.append(sum(C_nR[i:])/sum(C_nR))

    obs_FAR2.append(0)
    obs_HR2.append(0)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.fill_between(x=obs_FAR2, y1=obs_HR2, color='lightgray', alpha=.5)
    ax.plot(obs_FAR2, obs_HR2, 'ko-', linewidth=1.5, markersize=12,
            label='Observed')
    ax.set_title('Type 2 ROC curve')
    ax.set_ylabel('Type 2 P(correct)')
    ax.set_xlabel('Type 2 P(incorrect)')

    return ax
