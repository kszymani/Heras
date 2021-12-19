"""
Kod prezentuje implementacje funkcji aktywacji.
"""

from encrypted.array_utils import relinearize_array, refresh_array, encrypt_array, decrypt_array
from encrypted.maths import *


@debugger
def sigmoid(x, HE):
    coeffs = [1 / 2, 1 / 4, 0.0, -1 / 48, 0.0, 1 / 480, 0.0, -17 / 80640]
    res = evaluate_poly(x, coeffs, HE)
    return res


@debugger
def sigmoid_deriv(x, HE):
    coeffs = [1 / 4, 0.0, -1 / 16, 0.0, 1 / 96, 0.0, -17 / 11520]
    res = evaluate_poly(x, coeffs, HE)
    return res


@debugger
def relu(x, HE):
    sqr = x ** 2
    relinearize_array(sqr, HE)
    res = (x + sqrt(sqr, HE)) / 2.0
    return res


@debugger
def relu_deriv(x, HE):
    res = (sign(x, HE) + 1.0) / 2.0
    return res


@debugger
def square(x, HE):
    res = x ** 2
    relinearize_array(res, HE)
    return res


@debugger
def square_deriv(x, HE):
    res = x * 2.0
    return res


@debugger
def polynomial(x, HE):
    res = x ** 2 + x
    relinearize_array(res, HE)
    return res


@debugger
def polynomial_deriv(x, HE):
    return x * 2.0 + 1.0
