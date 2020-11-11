# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pandas as pd
import numpy as np
import unittest
import pytest
from metadPy.sdt import scores, rates, dprime, criterion, metad_MLE, roc_auc
from unittest import TestCase

data = pd.DataFrame(
    {
        "signal": np.concatenate(
            (np.ones(20), np.ones(5), np.zeros(10), np.zeros(15))
        ).astype(bool),
        "responses": np.concatenate(
            (np.ones(20), np.zeros(5), np.ones(10), np.zeros(15))
        ).astype(bool),
    }
)


class Testsdt(TestCase):
    def test_scores(self):
        """Test scores function"""
        assert (20, 5, 10, 15) == scores(data=data)
        assert (20, 5, 10, 15) == scores(
            signal=data.signal.to_numpy(), responses=data.responses.to_numpy()
        )
        with pytest.raises(ValueError):
            scores(data=None, signal=None, responses=None)

    def test_rates(self):
        """Test rates function"""
        assert (0.8, 0.4) == rates(20, 5, 10, 15)
        rates(0, 5, 0, 15)
        rates(5, 5, 5, 5)

    def test_dprime(self):
        """Test d prime function"""
        assert 1.095 == round(dprime(0.8, 0.4), 3)

    def test_criterion(self):
        """Test criterion function"""
        assert 0.294 == -round(criterion(0.8, 0.4), 3)

    def test_metad_MLE(self):
        """Test fit_meta_d_MLE function"""
        nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
        nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
        fit = metad_MLE(nR_S1, nR_S2)
        assert round(fit["meta_da"], 3) == 1.634
        fit["t2ca_rS1"]
        with pytest.raises(ValueError):
            fit = metad_MLE(np.zeros(7), nR_S2)
        with pytest.raises(ValueError):
            fit = metad_MLE(nR_S1[:1], nR_S2)

    def test_roc_auc(self):
        """Test roc_auc function"""
        nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
        nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
        auc = roc_auc(nR_S1, nR_S2)
        assert round(auc, 3) == 0.728


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
