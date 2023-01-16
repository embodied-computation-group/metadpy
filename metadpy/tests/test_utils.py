# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from metadpy.utils import (
    discreteRatings,
    pairedResponseSimulation,
    ratings2df,
    responseSimulation,
    trials2counts,
    trialSimulation,
    type2_SDT_simuation,
)

ratings = np.array(
    [
        96.0,
        98.0,
        95.0,
        90.0,
        32.0,
        58.0,
        77.0,
        6.0,
        78.0,
        78.0,
        62.0,
        60.0,
        38.0,
        12.0,
        63.0,
        18.0,
        15.0,
        13.0,
        49.0,
        26.0,
        2.0,
        38.0,
        60.0,
        23.0,
        25.0,
        39.0,
        22.0,
        33.0,
        32.0,
        27.0,
        40.0,
        13.0,
        35.0,
        16.0,
        35.0,
        73.0,
        50.0,
        3.0,
        40.0,
        0.0,
        34.0,
        47.0,
        52.0,
        0.0,
        0.0,
        0.0,
        25.0,
        1.0,
        16.0,
        37.0,
        59.0,
        20.0,
        25.0,
        23.0,
        45.0,
        22.0,
        28.0,
        62.0,
        61.0,
        69.0,
        20.0,
        75.0,
        10.0,
        18.0,
        61.0,
        27.0,
        63.0,
        22.0,
        54.0,
        30.0,
        36.0,
        66.0,
        14.0,
        2.0,
        53.0,
        58.0,
        88.0,
        23.0,
        77.0,
        54.0,
    ]
)


class Testsdt(TestCase):
    def test_trials2counts(self):
        """Test trials2counts function"""
        df = pd.DataFrame(
            {
                "Stimuli": [0, 1, 0, 0, 1, 1, 1, 1],
                "Accuracy": [0, 1, 1, 1, 0, 0, 1, 1],
                "Confidence": [1, 2, 3, 4, 4, 3, 2, 1],
                "nRatings": 4,
            }
        )
        nR_S1, nR_S2 = trials2counts(
            data=df,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            nRatings=4,
        )
        assert (nR_S1 == np.array([1, 1, 0, 0, 1, 0, 0, 0])).all()
        assert (nR_S2 == np.array([1, 1, 0, 0, 1, 2, 0, 0])).all()

        nR_S1, nR_S2 = df.trials2counts()
        assert (nR_S1 == np.array([1, 1, 0, 0, 1, 0, 0, 0])).all()
        assert (nR_S2 == np.array([1, 1, 0, 0, 1, 2, 0, 0])).all()

        with pytest.raises(ValueError):
            nR_S1, nR_S2 = trials2counts(
                data="error",
                nRatings=4,
            )
        with pytest.raises(ValueError):
            nR_S1, nR_S2 = trials2counts(
                stimuli=[0, 1, 0, 0, 1, 1, 1, 1],
                accuracy=[0, 1, 1, 1, 0, 1, 1],
                confidence=[1, 2, 3, 4, 4, 3, 2],
                nRatings=4,
            )
        nR_S1, nR_S2 = trials2counts(
            stimuli=[0, 1, 0, 0, 1, 1, 1, 1],
            accuracy=[0, 1, 1, 1, 0, 0, 1, 1],
            confidence=[1, 2, 3, 4, 4, 3, 2, 1],
            nRatings=4,
            padding=True,
        )
        assert (
            nR_S1 == np.array([1.125, 1.125, 0.125, 0.125, 1.125, 0.125, 0.125, 0.125])
        ).all()
        assert (
            nR_S2 == np.array([0.125, 0.125, 0.125, 0.125, 1.125, 2.125, 1.125, 1.125])
        ).all()
        nR_S1, nR_S2 = trials2counts(
            stimuli=np.array([0, 1, 0, 0, 1, 1, 1, 1]),
            accuracy=np.array([0, 1, 1, 1, 0, 0, 1, 1]),
            confidence=np.array([1, 2, 3, 4, 4, 3, 2, 1]),
            nRatings=4,
        )
        assert (nR_S1 == np.array([1, 1, 0, 0, 1, 0, 0, 0])).all()
        assert (nR_S2 == np.array([1, 1, 0, 0, 1, 2, 0, 0])).all()

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
        assert np.all(counts == np.array([51, 10, 9, 10]))

        ratingsHigh = ratings.copy()
        ratingsHigh[np.where(ratings > 50)[0]] = 99
        responseConf, out = discreteRatings(ratingsHigh, nbins=4)
        unique, counts = np.unique(responseConf, return_counts=True)
        assert np.all(counts == np.array([17, 17, 18, 28]))

    def test_trialSimulation(self):
        """Test trialSimulation function"""
        simulation_df = trialSimulation(
            d=1,
            metad=1,
            c=0,
            nRatings=4,
            nTrials=500,
        )
        assert isinstance(simulation_df, pd.DataFrame)
        assert len(simulation_df) == 500
        nR_S1, nR_S2 = simulation_df.trials2counts()
        assert sum(nR_S1) == sum(nR_S2) == 250

    def test_responseSimulation(self):
        """Test responseSimulation function"""
        # Single subject
        simulation_df = responseSimulation(d=1, metad=2, c=0, nRatings=4, nTrials=500)
        assert isinstance(simulation_df, pd.DataFrame)
        assert len(simulation_df) == 500
        nR_S1, nR_S2 = simulation_df.trials2counts()
        assert sum(nR_S1) == sum(nR_S2) == 250

        # Group of subjects
        simulation_df = responseSimulation(
            d=1, metad=2, c=0, nRatings=4, nTrials=500, nSubjects=10
        )
        assert isinstance(simulation_df, pd.DataFrame)
        assert simulation_df["Subject"].nunique() == 10
        nR_S1, nR_S2 = simulation_df.trials2counts()
        assert sum(nR_S1) == sum(nR_S2) == 2500

    def test_pairedResponseSimulation(self):
        """Test responseSimulation function"""
        simulation_df = pairedResponseSimulation()

        assert isinstance(simulation_df, pd.DataFrame)
        assert simulation_df["Subject"].nunique() == 20
        nR_S1, nR_S2 = simulation_df.trials2counts()
        assert sum(nR_S1) == sum(nR_S2) == 10000

    def test_type2_SDT_simuation(self):
        """Test responseSimulation function"""
        nR_S1, nR_S2 = type2_SDT_simuation(d=1, noise=0.2, c=0, nRatings=4, nTrials=500)
        assert len(nR_S1) == len(nR_S2) == 8
        nR_S1, nR_S2 = type2_SDT_simuation(
            d=1, noise=[0.2, 0.8], c=0, nRatings=4, nTrials=500
        )

    def test_ratings2df(self):
        """Test ratings2df function"""
        nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
        nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
        df = ratings2df(nR_S1, nR_S2)
        assert len(df) == sum(nR_S2) * 2
        assert df.Accuracy.sum() == (sum(nR_S1[:4]) + sum(nR_S2[4:]))

        # Test compatibility between ratings2df and trials2counts
        nR_S1bis, nR_S2bis = trials2counts(data=df, nRatings=4)
        assert np.all(nR_S1 == nR_S1bis)
        assert np.all(nR_S2 == nR_S2bis)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
