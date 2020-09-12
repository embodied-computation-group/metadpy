# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
import pytest
import numpy as np
from metadPy.utils import trials2counts, discreteRatings, responseSimulation,\
    type2_SDT_simuation, ratings2df
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
        nR_S1, nR_S2 = trials2counts(stimID=[0, 1, 0, 0, 1, 1, 1, 1],
                                     response=[0, 1, 1, 1, 0, 0, 1, 1],
                                     rating=[1, 2, 3, 4, 4, 3, 2, 1],
                                     nRatings=4)
        assert nR_S1 == [0, 0, 0, 1, 0, 0, 1, 1]
        assert nR_S2 == [1, 1, 0, 0, 1, 2, 0, 0]
        with pytest.raises(ValueError):
            nR_S1, nR_S2 = trials2counts(stimID=[0, 1, 0, 0, 1, 1, 1, 1],
                                         response=[0, 1, 1, 1, 0, 1, 1],
                                         rating=[1, 2, 3, 4, 4, 3, 2],
                                         nRatings=4)
        nR_S1, nR_S2 = trials2counts(stimID=[0, 1, 0, 0, 1, 1, 1, 1],
                                     response=[0, 1, 1, 1, 0, 0, 1, 1],
                                     rating=[1, 2, 3, 4, 4, 3, 2, 1],
                                     nRatings=4, padCells=True)

    def test_discreteRatings(self):
        """Test trials2counts function"""
        with pytest.raises(ValueError):
            discreteRatings([1, 1, 1, 1, 1], nbins=4)
        responseConf, out = discreteRatings(ratings, nbins=4)
        unique, counts = np.unique(responseConf, return_counts=True)
        assert np.all(counts == np.array([19, 20, 20, 21]))

        ratingsLow = ratings.copy()
        ratingsLow[np.where(ratings < 50)[0]] = 1
        responseConf, out = discreteRatings(ratingsLow, nbins=4)
        unique, counts = np.unique(responseConf, return_counts=True)
        assert np.all(counts == np.array([51, 10,  9, 10]))

        ratingsHigh = ratings.copy()
        ratingsHigh[np.where(ratings > 50)[0]] = 99
        responseConf, out = discreteRatings(ratingsHigh, nbins=4)
        unique, counts = np.unique(responseConf, return_counts=True)
        assert np.all(counts == np.array([17, 17, 18, 28]))

    def test_responseSimulation(self):
        """Test responseSimulation function"""
        nR_S1, nR_S2 = responseSimulation(
            d=1, metad=2, c=0, nRatings=4, nTrials=500)
        assert len(nR_S1) == len(nR_S2) == 8
        assert sum(nR_S1) == sum(nR_S2) == 250

    def test_type2_SDT_simuation(self):
        """Test responseSimulation function"""
        nR_S1, nR_S2 = type2_SDT_simuation(
            d=1, noise=.2, c=0, nRatings=4, nTrials=500)
        assert len(nR_S1) == len(nR_S2) == 8
        nR_S1, nR_S2 = type2_SDT_simuation(
            d=1, noise=[.2, .8], c=0, nRatings=4, nTrials=500)

    def test_ratings2df(self):
        """Test ratings2df function"""
        nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
        nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
        df = ratings2df(nR_S1, nR_S2)
        assert len(df) == sum(nR_S2)*2
        assert df.Accuracy.sum() == (sum(nR_S1[:4])+sum(nR_S2[:4]))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
