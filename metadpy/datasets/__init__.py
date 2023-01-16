# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

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
    return pd.read_csv(
        (
            "https://raw.githubusercontent.com/embodied-computation-group/metadpy/"
            "master/metadpy/datasets/rm.txt"
        )
    )
