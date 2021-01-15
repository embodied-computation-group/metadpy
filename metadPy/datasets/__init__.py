# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os.path as op

import pandas as pd


def load_dataset(dataset):
    """Load simulated dataset

    Parameters
    ----------
    dataset : str
        Which dataset. Defalut is `'rm'`

    Returns
    -------
    df : :py:class:`pandas.DataFrame`
        Pandas dataframe.
    """
    path = "https://github.com/LegrandNico/metadPy/raw/" "master/metadPy/datasets/"
    return pd.read_csv(op.join(path, "rm.txt"))
