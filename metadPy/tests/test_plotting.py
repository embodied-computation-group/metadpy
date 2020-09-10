# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
import pytest
import numpy as np
import matplotlib
from metadPy.plotting import plot_confidence
from metadPy.utils import responseSimulation
from unittest import TestCase


class Testsdt(TestCase):

    def test_trials2counts(self):
        """Test trials2counts function"""
        nR_S1, nR_S2 = responseSimulation(d=1, metad=2, c=0, nRatings=4,
                                          nTrials=500)
        ax = plot_confidence(nR_S1, nR_S2)
        assert isinstance(ax, matplotlib.axes.Axes)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
