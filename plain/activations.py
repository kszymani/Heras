"""
Kod prezentuje implementacje funkcji aktywacji.
"""

import numpy as np


def sigmoid_plain(x):
    res = 1 / (1 + np.exp(-x))
    return res


def sigmoid_deriv_plain(x):
    return sigmoid_plain(x) * (1 - sigmoid_plain(x))


def polynomial_plain(x):
    res = x ** 2 + x
    return res


def polynomial_deriv_plain(x):
    return x * 2.0 + 1.0


def relu_plain(x):
    return x * (x > 0)


def relu_deriv_plain(x):
    return 1 * (x > 0)


def square_plain(x):
    return x ** 2


def square_deriv_plain(x):
    return 2 * x
