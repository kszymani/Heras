import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def feed_forward(self, x):
        raise NotImplementedError

    def propagate_backward(self, error, lr):
        raise NotImplementedError


class Dense(Layer):
    def __repr__(self):
        return f"Dense({self.input_size}, {self.output_size})"

    def __init__(self, input_size, output_size, weights=None, bias=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        if weights is not None and bias is not None:
            self.weights = np.load(weights)
            self.bias = np.load(bias)
        else:
            start = -0.5
            end = 0.5
            random_weights = np.random.rand(input_size, output_size) * (end - start) + start
            random_bias = np.random.rand(1, output_size) * (end - start) + start
            self.weights = random_weights
            self.bias = random_bias

    def feed_forward(self, x):
        self.input = x
        output = np.dot(self.input, self.weights)
        output += self.bias
        return output

    def propagate_backward(self, output_error, lr):
        input_err = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= weights_error * lr
        self.bias -= output_error * lr
        return input_err


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def softmax(self, x):
        exps = np.exp(x)
        print("sum exps: ", np.sum(exps))
        inv = np.reciprocal(np.sum(exps))
        res = exps * inv
        return res

    def softmax_deriv(self):
        soft = self.output[0]
        s = soft.reshape(-1, 1)
        res = np.diagflat(s) - np.dot(s, s.T)
        return res

    def feed_forward(self, x):
        self.input = x
        self.output = self.softmax(x)
        return self.output

    def propagate_backward(self, error, lr):
        input_err = error @ self.softmax_deriv()
        return input_err


class Activation(Layer):
    def __init__(self, activation, activation_deriv):
        super().__init__()
        self.activation = activation
        self.activation_deriv = activation_deriv

    def feed_forward(self, x):
        self.input = x
        output = self.activation(self.input)
        return output

    def propagate_backward(self, error, lr):
        input_err = self.activation_deriv(self.input) * error
        return input_err
