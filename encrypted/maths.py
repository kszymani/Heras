"""
Kod implementuje metody przybliżeń funkcji, opartych o szeregi Taylora i metody iteracyjne.
"""

from Pyfhel import Pyfhel, PyCtxt
import numpy as np

from encrypted.array_utils import relinearize_array, refresh_array, decrypt_array, encrypt_array


def debugger(func):
    def wrapper(*args, **kwargs):
        dec = decrypt_array(*args).flatten()
        out = func(*args, **kwargs)
        fatal = False
        print("===================================")
        for e in zip(dec, decrypt_array(out, args[1]).flatten()):
            print("{:s}({:.6f}) = {:.6f}".format(func.__name__, e[0], e[1]))
            if np.abs(e[0]) > 10000 or np.abs(e[1]) > 10000:
                fatal = True
        print("===================================")
        if fatal:
            print("ERROR DETECTED")
            exit(420)
        return out
    return wrapper


@debugger
def sqrt(x, HE: Pyfhel, d=4):
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
    a = x * (-1.0) + 2.0
    b = x * (-1.0) + 1.0
    for i in range(d):
        b = b ** 2
        relinearize_array(b, HE)
        a *= b + 1
        relinearize_array(a, HE)
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
    for i in reversed(range(len(a))):
        result = (x * result) + a[i]
        relinearize_array(result, HE)
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
    coeffs = [-(363 / 140), 7, - 21 / 2, 35 / 3, - 35 / 4, 21 / 5, - 7 / 6, 1 / 7]
    return evaluate_poly(x, coeffs, HE)


@debugger
def exp(x, HE: Pyfhel):
    coeffs = [1.0, 1.0, 1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 720]
    return evaluate_poly(x, coeffs, HE)


def get_interval_id_for_sigmoid_from_client(x, HE: Pyfhel):
    args = decrypt_array(x, HE).flatten()
    res = []
    for a in args:
        if a < -6:
            res.append(0)
        elif -6 <= a < -2:
            res.append(1)
        elif -2 <= a < 2:
            res.append(2)
        elif 2 <= a < 6:
            res.append(3)
        else:
            res.append(4)
    return res


@debugger
def sigmoid_extended(x, HE: Pyfhel, coeffs_map=None):
    ids = get_interval_id_for_sigmoid_from_client(x, HE)
    result = np.empty(x.size, dtype=PyCtxt)
    i = 0
    for e in x.flatten():
        result[i] = evaluate_poly(np.array([e]), coeffs_map[ids[i]], HE)[0]
        i += 1
    return result.reshape(x.shape)


def get_map_sigmoid(HE):
    a0 = [0.000911051]  # for x < -6 sigmoid is almost flat line
    a1 = [0.61292, 0.450853, 0.141581, 0.0235304, 0.00205323, 0.0000747067]
    a2 = [1 / 2, 1 / 4, 0.0, -1 / 48, 0.0, 1 / 480, 0.0, -17 / 80640]
    a3 = [0.38708, 0.450853, -0.141581, 0.0235304, -0.00205323, 0.0000747067]
    a4 = [0.999089]
    a0 = encrypt_array(np.array(a0), HE)
    a1 = encrypt_array(np.array(a1), HE)
    a2 = encrypt_array(np.array(a2), HE)
    a3 = encrypt_array(np.array(a3), HE)
    a4 = encrypt_array(np.array(a4), HE)
    coeffs_map = {0: a0, 1: a1, 2: a2, 3: a3, 4: a4}
    return coeffs_map


@debugger
def sigmoid_extended_deriv(x, HE: Pyfhel, coeffs_map=None):
    ids = get_interval_id_for_sigmoid_from_client(x, HE)
    result = np.empty(x.size, dtype=PyCtxt)
    i = 0
    for e in x.flatten():
        result[i] = evaluate_poly(np.array([e]), coeffs_map[ids[i]], HE)[0]
        i += 1
    return result.reshape(x.shape)


def get_map_sigmoid_deriv(HE):
    a0 = [0.000910221]
    a1 = [0.450853, 0.283162, 0.0705913, 0.00821293, 0.000373533]
    a2 = [1 / 4, 0.0, -1 / 16, 0.0, 1 / 96, 0.0, -17 / 11520]
    a3 = [0.450853, -0.283162, 0.0705913, -0.00821293, 0.000373533]
    a4 = [0.000910221]
    a0 = encrypt_array(np.array(a0), HE)
    a1 = encrypt_array(np.array(a1), HE)
    a2 = encrypt_array(np.array(a2), HE)
    a3 = encrypt_array(np.array(a3), HE)
    a4 = encrypt_array(np.array(a4), HE)
    coeffs_map = {0: a0, 1: a1, 2: a2, 3: a3, 4: a4}
    return coeffs_map
