from Pyfhel import Pyfhel, PyCtxt
import numpy as np

from array_utils import relinearize_array, refresh_array, copy_array


def refresh(x, HE: Pyfhel):
    return HE.encryptFrac(HE.decryptFrac(x))


def sqrt(x, HE: Pyfhel, d=5):
    a = x
    b = x - 1.0
    for i in range(d):
        a *= b / -2.0 + 1.0
        relinearize_array(a, HE)
        sqr = b ** 2
        relinearize_array(sqr, HE)
        b = sqr * (b - 3.0) / 4.0
        relinearize_array(b, HE)

        a = refresh_array(a, HE)
        b = refresh_array(b, HE)
    return a


def reciprocal(x, HE: Pyfhel, d=5):
    a = x * (-1.0) + 2.0
    b = x * (-1.0) + 1.0
    for i in range(d):
        b = b ** 2
        relinearize_array(b, HE)
        a *= b + 1
        relinearize_array(a, HE)

        a = refresh_array(a, HE)
        b = refresh_array(b, HE)
    return a


def inverse_root(x, HE: Pyfhel, d=4):
    a = copy_array(x) + 0.5
    for i in range(d):
        sqr = a ** 2
        relinearize_array(sqr, HE)
        b = sqr * x
        relinearize_array(b, HE)
        b *= -0.5
        b += 1.5
        a *= b
        relinearize_array(a, HE)
        a = refresh_array(a, HE)
    return a


def sign(x, HE: Pyfhel):
    denom = x ** 2
    relinearize_array(denom, HE)
    res = x * inverse_root(denom, HE)
    relinearize_array(denom, HE)
    return res


def evaluate_poly(x, a, HE: Pyfhel):
    result = np.zeros(x.shape)
    for i in range(len(a) - 1, -1, -1):
        result = (x * result) + a[i]
        relinearize_array(result, HE)
    return result


def log(x, HE: Pyfhel):
    coeffs = [-137 / 60, 5.0, -5.0, 10 / 3, -5 / 4, 1 / 5]
    return evaluate_poly(x, coeffs, HE)
