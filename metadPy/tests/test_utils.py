# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
import pytest
from metadPy.utils import trials2counts
from unittest import TestCase


class Testsdt(TestCase):

    def test_trials2counts(self):
        """Test trials2counts function"""
        nR_S1, nR_S2 = trials2counts(stimID=[0, 1, 0, 0, 1, 1, 1, 1],
                                     response=[0, 1, 1, 1, 0, 0, 1, 1],
                                     rating=[1, 2, 3, 4, 4, 3, 2, 1],
                                     nRatings=4)
        assert nR_S1 == [0, 0, 0, 1, 0, 0, 1, 1]
        assert nR_S2 == [1, 1, 0, 0, 1, 2, 0, 0]


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
