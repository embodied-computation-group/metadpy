# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pandas as pd
import numpy as np
import unittest
import pytest
from metadPy.hierarchical import preprocess, hmetad_individual
from unittest import TestCase

nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
data = preprocess(nR_S1, nR_S2)

class Testsdt(TestCase):

    def test_preprocess(self):
        """Test preprocess function"""
        data = preprocess(nR_S1, nR_S2)
        assert round(data['d1'], 3) == 1.535
        assert round(data['c1']) == 0
        assert np.all(data['counts'] == np.array(
            [52, 32, 35, 37, 26, 12,  4,  2,  2, 5, 15, 22, 33, 38, 40, 45]))
        assert data['nratings'] == 4
        assert data['Tol'] == 1e-05
        assert data['FA'] == 44
        assert data['CR'] == 156
        assert data['M'] == 44
        assert data['H'] == 156
        assert data['N'] == 200
        assert data['S'] == 200

    def test_hmetad_individual(self):
        """Test hmetad_individual function"""
        results = hmetad_individual(data, draws=1000, chains=1)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
