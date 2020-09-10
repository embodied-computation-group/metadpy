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
        Where to draw the plot. Default is *None* (create a new figure).

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
    ax.set_ylabel('P$_{(Confidence|Precision, Accuracy)}$')
    ax.set_xlabel('Confidence rating')
    ax.set_xticks(range(1, nRratings+1))

    return ax
