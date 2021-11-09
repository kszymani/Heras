from encrypted.generate_context import restore_HE_from
from encrypted.array_utils import encrypt_array
from encrypted.network import Network
from encrypted.layers import Dense, Activation
from encrypted.activations import *
from encrypted.losses import *
from datasets import get_arrows

HE = restore_HE_from("../keys/light")

x_train, y_train, x_test, y_test, input_size = get_arrows(size=(8, 8), show=False)

print("Initializing network")
network = Network(seed=42069)
network.add(Dense(input_size, 5, HE))
network.add(Activation(square, square_deriv))
network.add(Dense(5, 1, HE))
network.add(Activation(sigmoid_squared, sigmoid_squared_deriv))
network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

epochs = 1
train_size = len(x_train)
test_size = len(x_test)
for j in range(epochs):
    print(f"\n\nEpoch {j+1}/{epochs}")
    for i in range(train_size):
        print("Sample {}/{}".format(i+1, len(x_train)))
        x_enc = encrypt_array(x_train[i], HE)
        y_enc = encrypt_array(y_train[i], HE)
        err = network.fit_sample(x_enc, y_enc, HE, lr=0.1)
        print("Loss ", HE.decryptFrac(err))

network.save_weights("arrow_weights", HE)
