import os
from datetime import datetime

from Pyfhel import Pyfhel

from encrypted.array_utils import decrypt_array
from encrypted.layers import Layer, Dense

import numpy as np


class Network:
    def __init__(self, seed=None):
        print("Building network")
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

    def predict_sample(self, sample, HE):
        start_time = datetime.now()
        output = sample
        for layer in self.layers:
            output = layer.feed_forward(output, HE)
        end_time = datetime.now()
        print('Sample predicting duration: {}'.format(end_time - start_time))
        return output

    def fit_sample(self, sample, target, HE: Pyfhel, lr):
        start_time = datetime.now()
        output = sample
        for layer in self.layers:
            output = layer.feed_forward(output, HE)
        err = self.loss(output, target, HE)

        error = self.loss_deriv(output, target, HE)
        for layer in reversed(self.layers):
            error = layer.propagate_backward(error, lr, HE)
        end_time = datetime.now()
        print('Sample fitting duration: {}'.format(end_time - start_time))
        return err

    def fit(self, samples, labels, HE: Pyfhel, epochs=1, lr=0.1):
        start_time = datetime.now()
        num_samples = len(samples)
        for e in range(epochs):
            print(f"\n\nEpoch {e + 1}/{epochs}")
            err = 0
            for s in range(num_samples):
                print(f"\nSample {s + 1}/{num_samples}")
                data = samples[s]
                label = labels[s]
                err += HE.decryptFrac(self.fit_sample(data, label, HE, lr))
            print(f"Error after epoch {e + 1}/{epochs}: {np.round(err / num_samples, 6)}")
        end_time = datetime.now()
        print('Total Duration: {}'.format(end_time - start_time))

    def save_weights_plain(self, folder_name, HE):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        i = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                np.save(f"{folder_name}/weights{i}.npy", decrypt_array(layer.weights, HE))
                np.save(f"{folder_name}/bias{i}.npy", decrypt_array(layer.bias, HE))
                i += 1
        print("Network parameters saved to folder {}".format(folder_name))

    def save_weights_enc(self, folder_name, HE):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        i = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                np.save(f"{folder_name}/weights{i}.npy", layer.weights)
                np.save(f"{folder_name}/bias{i}.npy", layer.bias)
                i += 1

