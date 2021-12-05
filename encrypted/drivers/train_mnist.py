import os
import random

from encrypted.array_utils import encrypt_array
from encrypted.generate_context import restore_HE_from
from encrypted.network import Network
from encrypted.layers import Dense, Activation
from encrypted.activations import *
from encrypted.losses import *
from datasets import get_mnist_data

HE = restore_HE_from("../keys/light")
# seed = random.randint(0, 10000)
seed = 6079
folder_name = "mnist_params"

x_train, y_train, x_test, y_test, input_size, test_values = get_mnist_data(seed=seed)

print("Initializing network with seed = ", seed)
network = Network(seed=seed)
network.add(Dense(input_size, 10, HE))
network.add(Activation(polynomial, polynomial_deriv))
network.add(Dense(10, 1, HE))
network.add(Activation(sigmoid, sigmoid_deriv))
network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

epochs = 5
train_size = len(x_train)
test_size = len(x_test)

try:
    for j in range(epochs):
        print(f"\n\nEpoch {j + 1}/{epochs}")
        for i in range(train_size):
            print("Sample {}/{}".format(i + 1, len(x_train)))
            x_enc = encrypt_array(x_train[i], HE)
            y_enc = encrypt_array(y_train[i], HE)
            err = network.fit_sample(x_enc, y_enc, HE, lr=0.01)
            print("Loss ", HE.decryptFrac(err))
        network.save_weights_plain(folder_name+str(j+1), HE)
except KeyboardInterrupt:
    print("Stopping and saving parameters")

network.save_weights_plain(folder_name, HE)
# 6079 seed
# 01:03 time
# refreshes per frame 31