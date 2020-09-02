# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pandas as pd
import numpy as np
import unittest
import pytest
from metadPy.sdt import scores, rates, dprime, criterion
from unittest import TestCase

data = pd.DataFrame({
    'signal': np.concatenate((np.ones(20), np.ones(5),
                              np.zeros(10), np.zeros(15))).astype(bool),
    'responses': np.concatenate((np.ones(20), np.zeros(5),
                                 np.ones(10), np.zeros(15))).astype(bool)})


class Testsdt(TestCase):

    def test_scores(self):
        """Test scores function"""
        assert (20, 5, 10, 15) == scores(data=data)
        assert (20, 5, 10, 15) == scores(signal=data.signal.to_numpy(),
                                         responses=data.responses.to_numpy())
        with pytest.raises(ValueError):
            scores(data=None, signal=None, responses=None)

    def test_rates(self):
        """Test rates function"""
        assert (0.8, 0.4) == rates(20, 5, 10, 15)

    def test_dprime(self):
        """Test d prime function"""
        assert 1.095 == round(dprime(0.8, 0.4), 3)

    def test_criterion(self):
        """Test criterion function"""
        assert .294 == -round(criterion(0.8, 0.4), 3)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
