from Pyfhel import Pyfhel, PyCtxt
import numpy as np

from array_utils import relinearize_array, refresh_array, decrypt_array


def debugger(func):
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        print(f"Output from {func.__name__}: ")
        for e in decrypt_array(out, args[1]):
            print("%.6f", e)
        print("===================================")
        return out
    return wrapper


def refresh(x, HE: Pyfhel):
    return HE.encryptFrac(HE.decryptFrac(x))


@debugger
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


@debugger
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


@debugger
def inverse_root(x, HE: Pyfhel, d=4):
    a = x + 0.5
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

@debugger
def sign(x, HE: Pyfhel):
    denom = x ** 2
    relinearize_array(denom, HE)
    res = x * inverse_root(denom, HE)
    relinearize_array(denom, HE)
    return res

@debugger
def evaluate_poly(x, a, HE: Pyfhel):
    result = np.zeros(x.shape)
    for i in range(len(a) - 1, -1, -1):
        result = (x * result) + a[i]
        relinearize_array(result, HE)
    return result


@debugger
def log(x, HE: Pyfhel):
    coeffs = [-137 / 60, 5.0, -5.0, 10 / 3, -5 / 4, 1 / 5]
    return evaluate_poly(x, coeffs, HE)
