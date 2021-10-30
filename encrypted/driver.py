from encrypted.array_utils import decrypt_array, encrypt_array
from encrypted.generate_context import restore_HE_from
from network import Network
from layers import Dense, Activation
from plain.datasets import get_mnist_data
from activations import *
from losses import *

HE = restore_HE_from("keypack")


network = Network(seed=345)

x_train = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.1]], [[1.0, 1.0]]])
y_train = np.array([[[0.0]], [[1.0]], [[1.0]], [[0.0]]])
print("encrypting arrays")
x_enc = encrypt_array(x_train, HE)
y_enc = encrypt_array(y_train, HE)

network.add(Dense(2, 3, HE))
network.add(Activation(square, square_deriv))
network.add(Dense(3, 1, HE))
network.add(Activation(sigmoid, sigmoid_deriv))

network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

network.fit(x_enc, y_enc, HE, epochs=1, lr=0.1)
network.save_weights("weights", HE)
out = network.predict(x_enc, HE)
result = decrypt_array(out, HE)
print(result)
