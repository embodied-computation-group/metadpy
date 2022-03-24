# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pymc as pm
import pytest

from metadPy import load_dataset
from metadPy.bayesian import extractParameters, hmetad


class Testsdt(TestCase):
    def test_extractParameters(self):
        """Test preprocess function"""
        nR_S1 = [52, 32, 35, 37, 26, 12, 4, 2]
        nR_S2 = [2, 5, 15, 22, 33, 38, 40, 45]
        data = extractParameters(nR_S1, nR_S2)
        assert round(data["d1"], 3) == 1.535
        assert round(data["c1"]) == 0
        assert np.all(
            data["counts"]
            == np.array([52, 32, 35, 37, 26, 12, 4, 2, 2, 5, 15, 22, 33, 38, 40, 45])
        )
        assert data["nratings"] == 4
        assert data["Tol"] == 1e-05
        assert data["FA"] == 44
        assert data["CR"] == 156
        assert data["M"] == 44
        assert data["H"] == 156
        assert data["N"] == 200
        assert data["S"] == 200

    def test_hmetad(self):
        """Test hmetad function"""
        group_df = load_dataset("rm")

        ####################
        # Test subject level
        ####################
        model, _ = hmetad(
            nR_S1=np.array([52, 32, 35, 37, 26, 12, 4, 2]),
            nR_S2=np.array([2, 5, 15, 22, 33, 38, 40, 45]),
            nRatings=4,
            sample_model=False,
            backend="numpyro",
        )
        assert callable(model)

        model, _ = hmetad(
            nR_S1=np.array([52, 32, 35, 37, 26, 12, 4, 2]),
            nR_S2=np.array([2, 5, 15, 22, 33, 38, 40, 45]),
            nRatings=4,
            sample_model=False,
            backend="pymc",
        )
        assert isinstance(model, pm.Model)

        with pytest.raises(ValueError):
            model = hmetad(
                data=None,
                nR_S1=None,
                nR_S2=None,
                nRatings=4,
                sample_model=False,
            )

        this_df = group_df[(group_df.Subject == 0) & (group_df.Condition == 0)]
        with pytest.raises(ValueError):
            model, _ = hmetad(
                data=this_df,
                nRatings=None,
                stimuli="Stimuli",
                accuracy="Accuracy",
                confidence="Confidence",
                sample_model=False,
            )

        model, _ = hmetad(
            data=this_df,
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            sample_model=False,
        )
        assert callable(model)

        # Force ratings discretization
        model, _ = hmetad(
            data=this_df,
            nRatings=3,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            sample_model=False,
        )
        assert callable(model)

        # Using Numpyro
        model, trace = hmetad(
            data=this_df,
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            backend="numpyro",
        )
        assert callable(model)
        assert isinstance(trace, dict)

        with pytest.raises(ValueError):
            model = hmetad(
                data=this_df,
                nRatings=4,
                stimuli="Stimuli",
                accuracy="Accuracy",
                confidence="Confidence",
                backend="inexistent_backend",
            )

        ####################
        # Test group level
        ####################
        this_df = group_df[group_df.Condition == 0]

        # Using Numpyro
        model, trace = hmetad(
            data=this_df,
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            subject="Subject",
        )

        assert callable(model)
        assert isinstance(trace, dict)

        #######################
        # Test repeated measure
        #######################

        # Using numpyro
        with pytest.raises(ValueError):
            model = hmetad(
                data=group_df,
                nRatings=4,
                stimuli="Stimuli",
                accuracy="Accuracy",
                confidence="Confidence",
                subject="Subject",
                within="Condition",
                backend="numpyro",
            )

        # Invalid backend
        with pytest.raises(ValueError):
            model = hmetad(
                data=group_df,
                nRatings=4,
                stimuli="Stimuli",
                accuracy="Accuracy",
                confidence="Confidence",
                subject="Subject",
                within="Condition",
                backend="inexistent_backend",
            )

        # Invalid design
        with pytest.raises(ValueError):
            model = hmetad(
                data=group_df,
                nRatings=4,
                stimuli="Stimuli",
                accuracy="Accuracy",
                confidence="Confidence",
                subject="Subject",
                between="Condition",
                within=None,
            )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
