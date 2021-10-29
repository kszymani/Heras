from Pyfhel import Pyfhel, PyCtxt
from array_utils import relinearize_array
from maths import sign, evaluate_poly, sqrt


def sigmoid(x, HE: Pyfhel):
    coeffs = [1 / 2, 1 / 4, 0.0, -1 / 48, 0.0, 1 / 480]
    return evaluate_poly(x, coeffs, HE)


def sigmoid_deriv(x, HE: Pyfhel):
    coeffs = [1 / 4, 0.0, -1 / 16, 0.0, 1 / 96]
    return evaluate_poly(x, coeffs, HE)


def relu(x, HE:Pyfhel):
    sqr = x**2
    relinearize_array(sqr, HE)
    return (x + sqrt(sqr, HE))/2.0


def relu_deriv(x, HE:Pyfhel):
    return (sign(x, HE) + 1.0)/2.0
