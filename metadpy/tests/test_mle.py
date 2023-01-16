# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pytest

from metadpy import load_dataset
from metadpy.mle import metad, norm_cdf


class Testsdt(TestCase):
    def test_norm_cdf(self):
        """Test norm_cdf function"""
        assert norm_cdf(0.75, 0, 1) == 0.7733726476231317

    def test_metad(self):
        """Test fit_meta_d_MLE function"""

        # From response-signal vectors
        # ----------------------------
        nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
        nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
        fit = metad(nR_S1=nR_S1, nR_S2=nR_S2, verbose=1)
        assert round(fit["dprime"][0], 3) == 1.535
        assert round(fit["meta_d"][0], 3) == 1.634
        assert round(fit["m_diff"][0], 3) == 0.099
        assert round(fit["m_ratio"][0], 3) == 1.064

        fit = metad(nR_S1=nR_S1, nR_S2=nR_S2, collapse=2, padding=False)

        # From dataframes
        # ---------------
        df = load_dataset("rm")

        # Subject level
        subject_fit = metad(
            data=df.copy(),
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
        )
        assert round(subject_fit["dprime"][0], 2) == 1.0
        assert round(subject_fit["meta_d"][0], 2) == 0.82
        assert round(subject_fit["m_ratio"][0], 2) == 0.82
        assert round(subject_fit["m_diff"][0], 2) == -0.18

        # Group level
        group_fit = metad(
            data=df.copy(),
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            subject="Subject",
        )
        assert group_fit["Subject"].nunique() == 20
        assert round(group_fit["meta_d"].mean(), 2) == 0.8
        assert round(group_fit["dprime"].mean(), 2) == 0.98
        assert round(group_fit["m_ratio"].mean(), 2) == 0.82
        assert round(group_fit["m_diff"].mean(), 2) == -0.17

        # Condition level 1
        condition_fit = metad(
            data=df.copy(),
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            subject="Subject",
            between="Condition",
        )
        assert condition_fit["Subject"].nunique() == 20
        assert condition_fit["Condition"].nunique() == 2
        assert round(condition_fit["meta_d"].mean(), 2) == 0.78
        assert round(condition_fit["dprime"].mean(), 2) == 0.96
        assert round(condition_fit["m_ratio"].mean(), 2) == 0.81
        assert round(condition_fit["m_diff"].mean(), 2) == -0.18

        # Condition level 2
        condition_fit_2 = metad(
            data=df.copy(),
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            subject="Subject",
            within="Condition",
        )
        assert condition_fit_2["Subject"].nunique() == 20
        assert condition_fit_2["Condition"].nunique() == 2
        assert round(condition_fit_2["meta_d"].mean(), 2) == 0.78
        assert round(condition_fit_2["dprime"].mean(), 2) == 0.96
        assert round(condition_fit_2["m_ratio"].mean(), 2) == 0.81
        assert round(condition_fit_2["m_diff"].mean(), 2) == -0.18

        # From arrays
        # -----------
        array_fit = metad(
            nRatings=4,
            stimuli=df.Stimuli.to_numpy(),
            accuracy=df.Accuracy.to_numpy(),
            confidence=df.Confidence.to_numpy(),
        )
        assert round(array_fit["meta_d"].mean(), 2) == 0.82
        assert round(array_fit["dprime"].mean(), 2) == 1.0
        assert round(array_fit["m_ratio"].mean(), 2) == 0.82
        assert round(array_fit["m_diff"].mean(), 2) == -0.18

        # When nRating not provided
        fit = metad(nR_S1=nR_S1, nR_S2=nR_S2, nRatings=None)

        # Using collapse instead of padding
        fit = metad(nR_S1=nR_S1, nR_S2=nR_S2, padding=None, collapse=2)

        with pytest.warns(UserWarning):
            fit = metad(nR_S1=np.zeros(8), nR_S2=np.zeros(8), padding=False)
        with pytest.raises(ValueError):
            fit = metad(nR_S1=np.zeros(7), nR_S2=nR_S2)
        with pytest.raises(ValueError):
            fit = metad(nR_S1=nR_S1[:1], nR_S2=nR_S2)
        with pytest.raises(ValueError):
            fit = metad(nR_S1=nR_S1, nR_S2=nR_S2, padding=True, collapse=2)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
