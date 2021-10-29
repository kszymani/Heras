import os

from Pyfhel import Pyfhel

from array_utils import decrypt_array
from layers import Layer, Dense

import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_deriv = None

    def add(self, layer: Layer):
        self.layers.append(layer)

    def set_loss(self, loss, loss_deriv):
        self.loss = loss
        self.loss_deriv = loss_deriv

    def predict(self, input_data, HE):
        print("Running inference")
        num_samples = len(input_data)
        result = []
        for s in range(num_samples):
            output = input_data[s]
            for layer in self.layers:
                output = layer.feed_forward(output, HE)
            result.append(output)
        return result

    def fit(self, input_data, labels, HE: Pyfhel, epochs=1, lr=0.1):
        num_samples = input_data[0]
        for e in range(epochs):
            err = 0
            for s in range(len(num_samples)):
                data = input_data[s]
                for layer in self.layers:
                    data = layer.feed_forward(data, HE)
                err += HE.decryptFrac(self.loss(data, labels, HE))

                error = self.loss_deriv(data, labels, HE)
                for layer in reversed(self.layers):
                    error = layer.propagate_backward(error, lr, HE)
            print(f"Error after epoch {e}/{epochs}: {err}")

    def save_weights(self, folder_name, HE):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        i = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                np.save(f"weights{i}.npy", decrypt_array(layer.weights, HE))
                np.save(f"bias{i}.npy", decrypt_array(layer.bias, HE))
                i += 1
