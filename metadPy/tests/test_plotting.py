# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import matplotlib
import pytest

from metadPy.plotting import plot_confidence, plot_roc
from metadPy.utils import responseSimulation
from metadPy.sdt import metad


class Testsdt(TestCase):
    def test_plot_confidence(self):
        """Test plot_confidence function"""
        nR_S1, nR_S2 = responseSimulation(d=1, metad=2, c=0, nRatings=4, nTrials=500)
        ax = plot_confidence(nR_S1, nR_S2)
        assert isinstance(ax, matplotlib.axes.Axes)
        with pytest.raises(ValueError):
            ax = plot_confidence(nR_S1[:-1], nR_S2)
        modelFit = metad(nR_S1, nR_S2)
        ax = plot_confidence(nR_S1, nR_S2, fitModel=modelFit)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_roc(self):
        """Test plot_roc function"""
        nR_S1, nR_S2 = responseSimulation(d=1, metad=2, c=0, nRatings=4, nTrials=500)
        ax = plot_roc(nR_S1, nR_S2)
        assert isinstance(ax, matplotlib.axes.Axes)
        modelFit = metad(nR_S1, nR_S2)
        ax = plot_roc(nR_S1, nR_S2, fitModel=modelFit)
        assert isinstance(ax[0], matplotlib.axes.Axes)
        assert isinstance(ax[1], matplotlib.axes.Axes)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
