from encrypted.array_utils import decrypt_array, encrypt_array
from encrypted.generate_context import restore_HE_from
from encrypted.network import Network
from encrypted.layers import Dense, Activation
from encrypted.activations import *
from encrypted.losses import *
import encrypted.array_utils as ref

HE = restore_HE_from("../keys/light")


network = Network()

x_train = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.1]], [[1.0, 1.0]]])
y_train = np.array([[[0.0]], [[1.0]], [[1.0]], [[0.0]]])
print("encrypting arrays")
x_enc = encrypt_array(x_train, HE)
y_enc = encrypt_array(y_train, HE)

network.add(Dense(2, 3, HE, weights='../weights/weights0.npy', bias='../weights/bias0.npy'))
network.add(Activation(square, square_deriv))
network.add(Dense(3, 1, HE, weights='../weights/weights1.npy', bias='../weights/bias1.npy'))
network.add(Activation(sigmoid, sigmoid_deriv))

network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

out = network.predict(x_enc, HE)
result = decrypt_array(out, HE)
print(result)

print("Total refreshes: ", ref.REFRESHES)
