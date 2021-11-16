
from datasets import get_mnist_data, get_mnist_data_categorical
from plain.network import Network
from plain.layers import Dense, Activation
from plain.activations import *
from plain.losses import *


network = Network(seed=42069)

x_train = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.1]], [[1.0, 1.0]]])
y_train = np.array([[[0.0]], [[1.0]], [[1.0]], [[0.0]]])
print("encrypting arrays")

network.add(Dense(2, 4))
network.add(Activation(square, square_deriv))
network.add(Dense(4, 1))
network.add(Activation(sigmoid, sigmoid_deriv))

network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

network.fit(x_train, y_train, epochs=50, lr=0.1)
out = network.predict(x_train)
out = np.round(out, 6)
print(out)

