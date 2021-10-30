import os

from Pyfhel import Pyfhel

from array_utils import decrypt_array
from layers import Layer, Dense

import numpy as np


class Network:
    def __init__(self, seed=None):
        self.layers = []
        self.loss = None
        self.loss_deriv = None
        if seed is not None:
            np.random.seed(seed)

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
            print(f"Sample {s + 1}/{num_samples}")
            output = input_data[s]
            for layer in self.layers:
                output = layer.feed_forward(output, HE)
            result.append(output)
        return np.array(result)

    def fit(self, input_data, labels, HE: Pyfhel, epochs=1, lr=0.1):
        num_samples = len(input_data)
        for e in range(epochs):
            print(f"Epoch {e+1}/{epochs}")
            err = 0
            for s in range(num_samples):
                print(f"Sample {s + 1}/{num_samples}")
                data = input_data[s]
                label = labels[s]
                for layer in self.layers:
                    data = layer.feed_forward(data, HE)
                    print("data noise after layer", layer, data[0, 0].noiseBudget)
                err += HE.decryptFrac(self.loss(data, label, HE))

                error = self.loss_deriv(data, label, HE)
                for layer in reversed(self.layers):
                    error = layer.propagate_backward(error, lr, HE)
                    print("error noise after layer", layer, error[0,0].noiseBudget)
            print(f"Error after epoch {e+1}/{epochs}: {np.round(err/num_samples, 6)}")

    def save_weights(self, folder_name, HE):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        i = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                np.save(f"{folder_name}/weights{i}.npy", decrypt_array(layer.weights, HE))
                np.save(f"{folder_name}/bias{i}.npy", decrypt_array(layer.bias, HE))
                i += 1
