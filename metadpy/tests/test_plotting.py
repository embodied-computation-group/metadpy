# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import matplotlib
import numpy as np
import pytest

from metadpy.mle import fit_metad
from metadpy.plotting import plot_confidence, plot_roc


class Testsdt(TestCase):
    def test_plot_confidence(self):
        """Test plot_confidence function"""
        nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
        nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
        ax = plot_confidence(nR_S1, nR_S2)
        assert isinstance(ax, matplotlib.axes.Axes)
        with pytest.raises(ValueError):
            ax = plot_confidence(nR_S1[:-1], nR_S2)
        fitModel = fit_metad(nR_S1=nR_S1, nR_S2=nR_S2, nRatings=4, nCriteria=7)
        ax = plot_confidence(nR_S1, nR_S2, fitModel=fitModel)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_roc(self):
        """Test plot_roc function"""
        nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
        nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
        ax = plot_roc(nR_S1, nR_S2)
        assert isinstance(ax, matplotlib.axes.Axes)
        fitModel = fit_metad(nR_S1=nR_S1, nR_S2=nR_S2, nRatings=4, nCriteria=7)
        ax = plot_roc(nR_S1, nR_S2, fitModel=fitModel)
        assert isinstance(ax[0], matplotlib.axes.Axes)
        assert isinstance(ax[1], matplotlib.axes.Axes)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
