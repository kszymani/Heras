"""
Kod implementuje binary crossentropy jako funkcję kosztu dla binarnej klasyfikacji.
"""

import numpy as np
from scipy.special import xlogy


def binary_crossentropy(p, y):
    res = xlogy(y, p) + xlogy(1-y, 1-p)
    return np.sum(res) / -y.size


def binary_crossentropy_deriv(p, y):
    res = (p - y) / (p - p ** 2)
    return res
