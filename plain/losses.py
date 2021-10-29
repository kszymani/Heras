
import numpy as np


def binary_crossentropy(p, y):
    res = y * np.log(p) + (1.0-y) * np.log(1.0-p)
    return np.sum(res)/-y.size


def binary_crossentropy_deriv(p, y):
    res = (p-y)/(p - p**2)
    return res
