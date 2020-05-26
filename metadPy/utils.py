# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from pymc3 import math


def cumulative_normal(x):
    return 0.5 + 0.5 * math.erf(x/math.sqrt(2))
