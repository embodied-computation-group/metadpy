# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pytest

from metadPy import load_dataset
from metadPy.mle import metad


class Testsdt(TestCase):
    def test_metad(self):
        """Test fit_meta_d_MLE function"""
        nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
        nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
        fit = metad(nR_S1=nR_S1, nR_S2=nR_S2)
        assert round(fit["dprime"][0], 3) == 1.535
        assert round(fit["metad"][0], 3) == 1.634
        assert round(fit["m_diff"][0], 3) == 0.099
        assert round(fit["m_ratio"][0], 3) == 1.064
        with pytest.raises(ValueError):
            fit = metad(nR_S1=np.zeros(7), nR_S2=nR_S2)
        with pytest.raises(ValueError):
            fit = metad(nR_S1=nR_S1[:1], nR_S2=nR_S2)
        with pytest.raises(ValueError):
            fit = metad(nR_S1=nR_S1, nR_S2=nR_S2, padding=True, collapse=2)
        df = load_dataset("rm")
        fit = metad(
            data=df[df.Subject == 0],
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            output_df=True,
            padding=False,
            collapse=2,
        )
        assert round(fit["metad"][0], 1) == 0.8


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
