from .datasets import load_dataset
from .hierarchical import extractParameters, hmetad
from .plotting import plot_confidence, plot_roc
from .sdt import criterion, dprime, fit_meta_d_logL, metad, rates, roc_auc, scores
from .utils import (
    discreteRatings,
    ratings2df,
    responseSimulation,
    trials2counts,
    type2_SDT_simuation,
)

__all__ = [
    "load_dataset",
    "hmetad",
    "extractParameters",
    "plot_confidence",
    "plot_roc",
    "scores",
    "rates",
    "dprime",
    "criterion",
    "fit_meta_d_logL",
    "metad",
    "roc_auc",
    "trials2counts",
    "discreteRatings",
    "responseSimulation",
    "type2_SDT_simuation",
    "ratings2df",
]

# Current version
__version__ = "0.0.1"
