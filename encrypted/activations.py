import numpy as np
from encrypted.array_utils import relinearize_array, refresh_array
from encrypted.maths import sign, evaluate_poly, sqrt, debugger, exp, reciprocal


@debugger
def sigmoid(x, HE):
    coeffs = [1 / 2, 1 / 4, 0.0, -1 / 48, 0.0, 1 / 480, 0.0, -17/80640]
    res = evaluate_poly(x, coeffs, HE)
    return res


@debugger
def sigmoid_deriv(x, HE):
    coeffs = [1 / 4, 0.0, -1 / 16, 0.0, 1 / 96, 0.0, -17/11520]
    res = evaluate_poly(x, coeffs, HE)
    return res


@debugger
def sigmoid_squared(x, HE):
    res = sigmoid(x, HE)
    res = res ** 2
    relinearize_array(res, HE)
    return res


@debugger
def sigmoid_squared_deriv(x, HE):
    sig = sigmoid(x, HE)
    res = sigmoid_deriv(x, HE)
    res *= sig * 2.0
    # res = refresh_array(der, HE)
    relinearize_array(res, HE)
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
def softmax(x, HE):
    exps = exp(x, HE)
    inv = reciprocal(np.sum(exps))
    res = exps * inv
    res = relinearize_array(res, HE)
    res = refresh_array(res, HE)
    return res


@debugger
def softmax_deriv(x, HE):
    soft = softmax(x, HE)
    outer = np.outer(soft, soft)
    relinearize_array(outer, HE)
    res = np.diag(soft.flatten()) - outer
    res = refresh_array(res, HE)
    return res
