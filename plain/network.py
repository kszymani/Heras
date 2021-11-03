import os

from plain.layers import Layer, Dense

import numpy as np


class Network:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.layers = []
        self.loss = None
        self.loss_deriv = None

    def add(self, layer: Layer):
        self.layers.append(layer)

    def set_loss(self, loss, loss_deriv):
        self.loss = loss
        self.loss_deriv = loss_deriv

    def predict(self, input_data):
        print("Running inference")
        num_samples = len(input_data)
        result = []
        for s in range(num_samples):
            output = input_data[s]
            for layer in self.layers:
                output = layer.feed_forward(output)
            result.append(output)
        return np.array(result)

    def predict_sample(self, sample):
        output = sample
        for layer in self.layers:
            output = layer.feed_forward(output)
        return output

    def fit_sample(self, data, label, lr):
        for layer in self.layers:
            data = layer.feed_forward(data)
        err = self.loss(data, label)
        error = self.loss_deriv(data, label)
        for layer in reversed(self.layers):
            error = layer.propagate_backward(error, lr)
        return err


    def fit(self, input_data, labels, epochs=1, lr=0.1):
        num_samples = len(input_data)
        for e in range(epochs):
            err = 0
            for s in range(num_samples):
                data = input_data[s]
                label = labels[s]
                err += self.fit_sample(data, label, lr)
            print(f"Error after epoch {e+1}/{epochs}: {np.round(err/num_samples, 6)}")

    def save_weights(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        i = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                np.save(f"weights{i}.npy", layer.weights)
                np.save(f"bias{i}.npy", layer.bias)
                i += 1
