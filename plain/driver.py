from plain.network import Network
from plain.layers import Dense, Activation
from plain.activations import *
from plain.losses import *

network = Network(seed=1234)

x_train = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.1]], [[1.0, 1.0]]])
y_train = np.array([[[0.0]], [[1.0]], [[1.0]], [[0.0]]])

network.add(Dense(2, 3))
network.add(Activation(relu, relu_deriv))
network.add(Dense(3, 1))
network.add(Activation(sigmoid, sigmoid_deriv))

network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

network.fit(x_train, y_train, epochs=500, lr=0.1)
print(network.predict(x_train))
