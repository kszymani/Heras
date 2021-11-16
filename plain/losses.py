import numpy as np


def mse(p, y):
    res = np.sum((y - p) ** 2) / y.size
    return res


def mse_deriv(p, y):
    res = 2 / y.size * (y - p)
    return res


def binary_crossentropy(p, y):
    res = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    return np.sum(res) / -y.size


def binary_crossentropy_deriv(p, y):
    res = (p - y) / (p - p ** 2)
    return res


def categorical_crossentropy(p, y):
    res = y * np.log(p)
    return np.sum(res) * (-1.0)


def categorical_crossentropy_deriv(p, y):
    res = p - y
    return res
