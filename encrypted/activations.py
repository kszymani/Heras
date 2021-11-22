
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
    # res = refresh_array(res, HE)
    relinearize_array(res, HE)
    return res


@debugger
def relu(x, HE):
    sqr = x ** 2
    relinearize_array(sqr, HE)
    res = (x + sqrt_taylor(sqr, HE)) / 2.0
    return res


@debugger
def relu_deriv(x, HE):
    res = (sign(x, HE) + 1.0) / 2.0
    return res


@debugger
def relu_client(x, HE):
    dec = decrypt_array(x, HE)
    res = encrypt_array(dec * (dec > 0), HE)
    return res


@debugger
def relu_client_deriv(x, HE):
    dec = decrypt_array(x, HE)
    res = encrypt_array(1.0 * (dec > 0), HE)
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
