from Pyfhel import Pyfhel, PyCtxt
import numpy as np

from array_utils import relinearize_array, refresh_array, decrypt_array


def debugger(func):
    def wrapper(*args, **kwargs):
        dec = decrypt_array(*args).flatten()
        print(f"input to {func.__name__}: ", dec)
        out = func(*args, **kwargs)
        fatal = False
        for e in zip(dec, decrypt_array(out, args[1]).flatten()):
            print("{:s}({:.6f}) = {:.6f}".format(func.__name__, e[0], e[1]))
            if np.abs(e[0]) > 100 or np.abs(e[1]) > 100:
                fatal = True
        print("===================================")
        if fatal:
            print("FATAL ERROR DETECTED")
            exit(420)
        return out

    return wrapper


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

        # a = refresh_array(a, HE)
        b = refresh_array(b, HE)
    return a


@debugger
def inverse_root(x, HE: Pyfhel, d=5):
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
    relinearize_array(res, HE)
    return res


def evaluate_poly(x, a, HE: Pyfhel):
    result = np.zeros(x.size, dtype=PyCtxt)
    for i in range(len(a) - 1, -1, -1):
        result = (x * result) + a[i]
        relinearize_array(result, HE)
        if result[0,0].noiseBudget < 400:
            print("POLY REFRESHING ARR")
            result = refresh_array(result, HE)
    return result


@debugger
def log(x, HE: Pyfhel):
    coeffs = [-137 / 60, 5.0, -5.0, 10 / 3, -5 / 4, 1 / 5]
    return evaluate_poly(x, coeffs, HE)
