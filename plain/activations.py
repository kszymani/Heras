import numpy as np


def sigmoid(x):
    # print("Input to sigmoid", x)
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * sigmoid(1 - x)


def sigmoid_squared(x):
    return sigmoid(x) ** 2


def sigmoid_squared_deriv(x):
    return 2 * np.exp(-x) / (1 + np.exp(-x)) ** 3


def relu(x):
    return x * (x > 0)


def relu_deriv(x):
    return 1 * (x > 0)


def square(x):
    return x ** 2


def square_deriv(x):
    return 2 * x

