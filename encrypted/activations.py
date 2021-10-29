from Pyfhel import Pyfhel, PyCtxt
from array_utils import relinearize_array, refresh_array
from maths import sign, evaluate_poly, sqrt, debugger


@debugger
def sigmoid(x, HE: Pyfhel):
    coeffs = [1 / 2, 1 / 4, 0.0, -1 / 48, 0.0, 1 / 480]
    res = evaluate_poly(x, coeffs, HE)
    return res


@debugger
def sigmoid_deriv(x, HE: Pyfhel):
    coeffs = [1 / 4, 0.0, -1 / 16, 0.0, 1 / 96]
    res = evaluate_poly(x, coeffs, HE)
    return res


@debugger
def relu(x, HE: Pyfhel):
    sqr = x ** 2
    relinearize_array(sqr, HE)
    res = (x + sqrt(sqr, HE)) / 2.0
    return res


@debugger
def relu_deriv(x, HE: Pyfhel):
    res = (sign(x, HE) + 1.0) / 2.0
    return res
