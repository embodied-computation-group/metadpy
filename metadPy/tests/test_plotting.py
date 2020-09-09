# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
import pytest
import numpy as np
import matplotlib
from metadPy.plotting import plot_confidence
from metadPy.utils import responseSimulation
from unittest import TestCase

ratings = np.array([
    96., 98., 95., 90., 32., 58., 77.,  6., 78., 78., 62., 60., 38.,
    12., 63., 18., 15., 13., 49., 26.,  2., 38., 60., 23., 25., 39.,
    22., 33., 32., 27., 40., 13., 35., 16., 35., 73., 50.,  3., 40.,
    0., 34., 47., 52.,  0.,  0.,  0., 25.,  1., 16., 37., 59., 20.,
    25., 23., 45., 22., 28., 62., 61., 69., 20., 75., 10., 18., 61.,
    27., 63., 22., 54., 30., 36., 66., 14.,  2., 53., 58., 88., 23.,
    77., 54.])


class Testsdt(TestCase):

    def test_trials2counts(self):
        """Test trials2counts function"""
        nR_S1, nR_S2 = responseSimulation(d=1, metad=2, c=0, nRatings=4,
                                          nTrials=500)
        fig, ax = plot_confidence(nR_S1, nR_S2)
        assert isinstance(ax, matplotlib.axes.Axes)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
