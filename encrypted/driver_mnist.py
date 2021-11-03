import os

from encrypted.generate_context import restore_HE_from
from encrypted.array_utils import decrypt_array, encrypt_array
from network import Network
from layers import Dense, Activation
from activations import *
from losses import *
from plain.datasets import get_mnist_data


def save_encrypted_images(samples, folder_name, HE):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for s in range(len(samples)):
        img = samples[s]
        enc = encrypt_array(img, HE)
        np.save("{}/{}.npy".format(folder_name, s), enc)


def load_encrypted_image(s, folder_name):
    arr = np.load("{}/{}.npy".format(folder_name, s), allow_pickle=True)
    return arr


HE = restore_HE_from("light")

x_train, y_train, x_test, y_test, input_size, test_values = get_mnist_data(seed=9876)

print("Initializing network")
network = Network(seed=5678)
network.add(Dense(input_size, 5, HE))
network.add(Activation(square, square_deriv))
network.add(Dense(5, 1, HE))
network.add(Activation(sigmoid_squared, sigmoid_squared_deriv))
network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

epochs = 2
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

network.save_weights("mnist_wieghts", HE)

correct_preds = 0
preds = 0
for i in range(test_size):
    x_enc = x_test[i]
    y = y_test[i][0]
    value = test_values[i][0]
    pred = decrypt_array(network.predict(x_enc, HE), HE)[0]

    p = np.round(pred)
    correct = p == y
    if correct:
        correct_preds += 1
    preds += 1
    print(f"{value} is {'even' if p == 0 else 'odd'} ({'GOOD' if correct else 'BAD'}) ({np.round(pred, 4)})")
print("Correct predictions: {}/{} ({:.2f}%)".format(correct_preds, preds, 100 * correct_preds / preds))
