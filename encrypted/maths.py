from Pyfhel import Pyfhel, PyCtxt
import numpy as np

from encrypted.array_utils import relinearize_array, refresh_array, decrypt_array

DEBUG = True


def debugger(func):
    def wrapper(*args, **kwargs):
        dec = decrypt_array(*args).flatten()
        out = func(*args, **kwargs)
        if DEBUG:
            fatal = False
            print("===================================")
            for e in zip(dec, decrypt_array(out, args[1]).flatten()):
                print("{:s}({:.6f}) = {:.6f}".format(func.__name__, e[0], e[1]))
                if np.abs(e[0]) > 100 or np.abs(e[1]) > 100:
                    # since we are usually dealing with small numbers, a big number randomly showing up is a good
                    # indicator that something went wrong
                    fatal = True
            print("===================================")
            if fatal:
                print("FATAL ERROR DETECTED")
                exit(420)
        return out

    return wrapper


@debugger
def sqrt(x, HE: Pyfhel, d=3):
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
def reciprocal(x, HE: Pyfhel, d=4):
    """Intended for smaller inputs x in (0, 2)"""
    a = x * (-1.0) + 2.0
    b = x * (-1.0) + 1.0
    for i in range(d):
        b = b ** 2
        relinearize_array(b, HE)
        a *= b + 1
        relinearize_array(a, HE)
        """because b approaches zero, it is necessary to frequently refresh the array to keep the noise as low as 
        possible in order to prevent it from overflowing the value """
        b = refresh_array(b, HE)
    return a


@debugger
def inverse_root(x, HE: Pyfhel, d=3):
    a = x + 0.5
    for i in range(d):
        sqr = a ** 2
        relinearize_array(sqr, HE)
        b = sqr * x
        relinearize_array(b, HE)
        b *= -0.5
        b += 1.5
        a *= b
        # relinearize_array(a, HE)
        a = refresh_array(a, HE)
    return a


@debugger
def sign(x, HE: Pyfhel):
    denom = x ** 2
    relinearize_array(denom, HE)
    res = x * inverse_sqrt_taylor(denom, HE)
    relinearize_array(res, HE)
    return res


def evaluate_poly(x, a, HE: Pyfhel):
    result = np.zeros(x.size, dtype=PyCtxt)
    for i in range(len(a) - 1, -1, -1):
        result = (x * result) + a[i]
        relinearize_array(result, HE)
        if i == len(a) // 2:
            result = refresh_array(result, HE)
    result = refresh_array(result, HE)
    return result


@debugger
def reciprocal_taylor(x, HE):
    coeefs = [8, - 28, 56, - 70, 56, - 28, + 8, - 1]
    return evaluate_poly(x, coeefs, HE)


@debugger
def sqrt_taylor(x, HE):
    coeefs = [429 / 2048, 3003 / 2048, - 3003 / 2048, 3003 / 2048, - 2145 / 2048, 1001 / 2048, - 273 / 2048, 33 / 2048]
    return evaluate_poly(x, coeefs, HE)


@debugger
def inverse_sqrt_taylor(x, HE):
    coeefs = [6435 / 2048, - 15015 / 2048, 27027 / 2048, - 32175 / 2048, 25025 / 2048, - 12285 / 2048,
              3465 / 2048, - 429 / 2048]
    return evaluate_poly(x, coeefs, HE)


@debugger
def log(x, HE: Pyfhel):
    coeffs = [-(363 / 140), 7, - 21 / 2, 35 / 3, - 35 / 4, 21 / 5, - 7 / 6, + 1 / 7]
    return evaluate_poly(x, coeffs, HE)


@debugger
def exp(x, HE: Pyfhel):
    coeffs = [1.0, 1.0, 1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 720]
    return evaluate_poly(x, coeffs, HE)
