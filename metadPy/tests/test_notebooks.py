# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
import unittest
from unittest import TestCase

import papermill as pm


class TestNotebooks(TestCase):
    def test_notebooks(self):
        """Test tutorial notebooks"""

        # Load tutorial notebooks from the GitHub repository
        url = "./notebooks/"
        for nb in [
            "1-What metacognition looks like.ipynb",
            "2-Fitting the model-MLE.ipynb",
            "Example 1 - Fitting MLE - Subject and group level.ipynb",
            "Example 2 - Fitting Bayesian - Subject level (numpyro).ipynb",
            "Example 3 - Fitting Bayesian - Subject level (pymc).ipynb",
            "QuickTour.ipynb"
        ]:
            pm.execute_notebook(url + nb, "./tmp.ipynb")
        os.remove("./tmp.ipynb")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
