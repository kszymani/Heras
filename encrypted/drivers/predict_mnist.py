import os

from encrypted.generate_context import restore_HE_from
from encrypted.array_utils import decrypt_array, encrypt_array
from encrypted.network import Network
from encrypted.layers import Dense, Activation
from encrypted.activations import *
from encrypted.losses import *
from datasets import get_mnist_data

HE = restore_HE_from("../keys/light")


x_train, y_train, x_test, y_test, input_size, test_values = get_mnist_data(seed=9876)

print("Initializing network")
network = Network(seed=5678)
network.add(Dense(input_size, 5, HE, weights='mnist_weights/weights0.npy', bias='mnist_weights/bias0.npy'))
network.add(Activation(square, square_deriv))
network.add(Dense(5, 1, HE, weights='mnist_weights/weights1.npy', bias='mnist_weights/bias1.npy'))
network.add(Activation(sigmoid_squared, sigmoid_squared_deriv))
network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

train_size = len(x_train)
test_size = len(x_test)

correct_preds = 0
preds = 0
for i in range(test_size):
    print(f"Predicting sample {i + 1}/{test_size}")
    x_enc = encrypt_array(x_test[i], HE)
    y = y_test[i][0]
    value = test_values[i][0]
    pred = decrypt_array(network.predict_sample(x_enc, HE), HE)[0, 0]

    p = np.round(pred)
    correct = p == y
    if correct:
        correct_preds += 1
    preds += 1
    print(f"{value} is {'even' if p == 0 else 'odd'} ({'GOOD' if correct else 'BAD'}) ({np.round(pred, 4)})")
print("Correct predictions: {}/{} ({:.2f}%)".format(correct_preds, preds, 100 * correct_preds / preds))
