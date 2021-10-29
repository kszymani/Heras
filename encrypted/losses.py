import numpy as np

from array_utils import relinearize_array, refresh_array
from maths import log, reciprocal


def binary_crossentropy(p, y, HE):
    res = y * log(p, HE) + (y * -1.0 + 1.0) * log(p * -1.0 + 1.0, HE)
    relinearize_array(res, HE)
    return np.sum(res) / -y.size


def binary_crossentropy_deriv(p, y, HE):
    denom = p ** 2
    relinearize_array(denom, HE)
    denom *= -1.0
    inv = reciprocal(denom + p, HE)
    res = (p - y) * inv
    relinearize_array(res, HE)
    if res[0,0].noiseBudget < 300:
        res = refresh_array(res, HE)
    return res
