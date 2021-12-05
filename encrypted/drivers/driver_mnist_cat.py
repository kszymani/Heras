import os

from encrypted.generate_context import restore_HE_from
from encrypted.array_utils import encrypt_array
from encrypted.network import Network
from encrypted.layers import Dense, Activation
from encrypted.activations import *
from encrypted.losses import *
from datasets import get_mnist_data_categorical

HE = restore_HE_from("../keys/light")

x_train, y_train, x_test, y_test, input_size, test_values = get_mnist_data_categorical(seed=372661)
# train_length = 393
# x_train = x_train[:train_length]
# y_train = y_train[:train_length]

network = Network(seed=8854524)

network.add(Dense(input_size, 32, HE))
network.add(Activation(polynomial, polynomial_deriv))
network.add(Dense(32, 16, HE))
network.add(Activation(polynomial, polynomial_deriv))
network.add(Dense(16, 10, HE))
network.add(Activation(sigmoid, sigmoid_deriv))
network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

epochs = 1
train_size = len(x_train)
test_size = len(x_test)
try:
    for j in range(epochs):
        print(f"\n\nEpoch {j+1}/{epochs}")
        for i in range(train_size):
            print("Sample {}/{}".format(i+1, len(x_train)))
            x_enc = encrypt_array(x_train[i], HE)
            y_enc = encrypt_array(y_train[i], HE)
            err = network.fit_sample(x_enc, y_enc, HE, lr=0.05)
            print("Loss ", HE.decryptFrac(err))
except KeyboardInterrupt:
    network.save_weights_plain("mnist_weights3", HE)

test_size = len(x_test)

correct_preds = 0
preds = 0
for i in range(test_size):
    print(f"Predicting sample {i + 1}/{test_size}")
    x_enc = encrypt_array(x_test[i], HE)
    y = y_test[i][0]
    pred = decrypt_array(network.predict_sample(x_enc, HE), HE)[0]

    p = np.argmax(pred)
    digit = np.argmax(y)
    correct = p == digit
    if correct:
        correct_preds += 1
    preds += 1
    print(f" predicted {p}, expexted: ({digit})")
print("Correct predictions: {}/{} ({:.2f}%)".format(correct_preds, preds, 100 * correct_preds / preds))

preds= 0
correct_preds = 0


