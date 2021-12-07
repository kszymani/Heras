import numpy as np


def sigmoid_plain(x):
    # if np.any(np.abs(x) > 2.1):
    #     print("out of bounds for sigmoid approximation")
    #     print(x)
    #     exit(69)
    res = 1 / (1 + np.exp(-x))
    return res


def sigmoid_deriv_plain(x):
    return sigmoid_plain(x) * (1 - sigmoid_plain(x))


def sigmoid_scaled_plain(x):
    if np.any(np.abs(x) > 6.1):
        print("out of bounds for sigmoid approximation")
        print(x)
        exit(69)
    alpha = 1 / 3
    res = 1 / (1 + np.exp(-alpha * x))
    return res


def sigmoid_scaled_deriv_plain(x):
    alpha = 1 / 3
    return alpha * sigmoid_scaled_plain(x) * (1 - sigmoid_scaled_plain(x))


def polynomial_plain(x):
    res = x ** 2 + x
    return res


def polynomial_deriv_plain(x):
    return x * 2.0 + 1.0


def sigmoid_squared_plain(x):
    return sigmoid_plain(x) ** 2


def sigmoid_squared_deriv_plain(x):
    return 2 * np.exp(-x) / (1 + np.exp(-x)) ** 3


def relu_plain(x):
    return x * (x > 0)


def relu_deriv_plain(x):
    return 1 * (x > 0)


def square_plain(x):
    return x ** 2


def square_deriv_plain(x):
    return 2 * x
