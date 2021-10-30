import numpy as np

from array_utils import encrypt_array, relinearize_array, refresh_array, decrypt_array

BUDGET = 300


class Layer:
    def __init__(self, ):
        self.input = None
        self.output = None

    def feed_forward(self, x, HE):
        raise NotImplementedError

    def propagate_backward(self, error, lr, HE):
        raise NotImplementedError


class Dense(Layer):
    def __repr__(self):
        return f"Dense({self.input_size}, {self.output_size})"

    def __init__(self, input_size, output_size, HE, weights=None, bias=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        if weights is not None and bias is not None:
            self.weights = encrypt_array(np.load(weights, allow_pickle=True), HE)
            self.bias = encrypt_array(np.load(bias, allow_pickle=True), HE)
        else:
            random_weights = np.random.rand(input_size, output_size) - 0.5
            random_bias = np.random.rand(1, output_size) - 0.5
            self.weights = encrypt_array(random_weights, HE)
            self.bias = encrypt_array(random_bias, HE)

    def feed_forward(self, x, HE):
        self.input = x
        output = np.dot(self.input, self.weights)
        relinearize_array(output, HE)
        output += self.bias
        output = refresh_array(output, HE)
        return output

    def propagate_backward(self, output_err, lr, HE):
        input_err = np.dot(output_err, self.weights.T)
        relinearize_array(input_err, HE)
        weights_err = np.dot(self.input.T, output_err)
        relinearize_array(weights_err, HE)

        self.weights -= weights_err * lr
        self.bias -= output_err * lr
        print("self.weights noise in layer ", self, self.weights)
        if self.weights[0, 0].noiseBudget < BUDGET:
            self.weights = refresh_array(self.weights, HE)
        print("self.bias noise in layer ", self, self.bias)
        if self.bias[0, 0].noiseBudget < BUDGET:
            self.bias = refresh_array(self.bias, HE)
        input_err = refresh_array(input_err, HE)
        return input_err


class Activation(Layer):
    def __repr__(self):
        return f"Activation({self.activation.__name__}, {self.activation_deriv.__name__})"

    def __init__(self, activation, activation_deriv):
        super().__init__()
        self.activation = activation
        self.activation_deriv = activation_deriv

    def feed_forward(self, x, HE):
        self.input = refresh_array(x, HE)
        output = self.activation(self.input, HE)
        return output

    def propagate_backward(self, output_err, lr, HE):
        r = self.activation_deriv(self.input, HE)
        input_err = r * output_err
        relinearize_array(input_err, HE)
        input_err = refresh_array(input_err, HE)
        return input_err
