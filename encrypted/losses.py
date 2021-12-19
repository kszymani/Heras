"""
Kod implementuje binary crossentropy jako funkcję kosztu dla binarnej klasyfikacji.
"""

import numpy as np

from encrypted.array_utils import relinearize_array, refresh_array
from encrypted.maths import log, reciprocal


def binary_crossentropy(p, y, HE):
    res = y * log(p, HE) + (y * -1.0 + 1.0) * log(p * -1.0 + 1.0, HE)
    relinearize_array(res, HE)
    return np.sum(res) / -y.size


def binary_crossentropy_deriv(p, y, HE):
    denom = p ** 2
    relinearize_array(denom, HE)
    inv = reciprocal(p - denom, HE)
    res = (p - y) * inv
    relinearize_array(res, HE)
    res = refresh_array(res, HE)
    return res
