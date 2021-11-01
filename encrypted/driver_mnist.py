from encrypted.generate_context import restore_HE_from
from encrypted.array_utils import decrypt_array, encrypt_array
from network import Network
from layers import Dense, Activation
from activations import *
from losses import *
from plain.datasets import get_mnist_data

HE = restore_HE_from("keypack")


x_train, y_train, x_test, y_test, input_size, test_values = get_mnist_data(seed=9876)
network = Network(seed=5678)

network.add(Dense(input_size, 10, HE))
network.add(Activation(square, square_deriv))
network.add(Dense(10, 1, HE))
network.add(Activation(sigmoid, sigmoid_deriv))

network.set_loss(binary_crossentropy, binary_crossentropy_deriv)


for i in range(10):
    print("Batch ", i)
    start = i*10
    end = 10*(i+1)
    x_enc = encrypt_array(x_train[start:end], HE)
    y_enc = encrypt_array(y_train[start:end], HE)
    network.fit(x_enc, y_enc, HE, epochs=1, lr=0.1)

network.save_weights("mnist_wieghts", HE)

correct_preds = 0
preds = 0
for i in range(4):
    print("Test Batch ", i)
    start = i*10
    end = 10*(i+1)
    y = y_test[start:end]
    val = test_values[start:end]
    x_enc = encrypt_array(x_test[start:end], HE)
    y_enc = encrypt_array(y, HE)
    pred = decrypt_array(network.predict(x_test, HE), HE)

    correct_preds = 0
    for e in zip(pred.flatten(), y.flatten(), val.flatten()):
        p = np.round(e[0])
        correct = p == e[1]
        if correct:
            correct_preds += 1
        preds += 1
        print(f"{e[2]} is {'even' if p == 0 else 'odd'} ({'GOOD' if correct else 'BAD'}) ({np.round(e[0], 4)})")
print("Correct predictions: {}/{} ({:.2f}%)".format(correct_preds, preds, 100 * correct_preds / preds))