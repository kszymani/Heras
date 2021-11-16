from datasets import *
from plain.network import Network
from plain.layers import Dense, Activation, SoftmaxLayer
from plain.activations import *
from plain.losses import *

x_train, y_train, x_test, y_test, input_size, test_values = get_mnist_data_categorical(seed=9876)

network = Network(seed=5678)

network.add(Dense(input_size, 15))
network.add(Activation(relu, relu_deriv))
network.add(Dense(15, 10))
network.add(SoftmaxLayer())  # Activation(sigmoid, sigmoid_deriv)
network.set_loss(categorical_crossentropy, categorical_crossentropy_deriv)

network.fit(x_train, y_train, epochs=50, lr=0.1)
pred = network.predict(x_test)
correct_preds = 0
for e in zip(pred, y_test, test_values):
    p = np.argmax(e[0][0])
    probablity = e[0][0][p]
    y = e[2][0]
    correct = p == y
    if correct:
        correct_preds += 1
    # print("predicted {}, actual: {} ({:.6f})".format(p, y, probablity))

print("Correct predictions: {}/{} ({:.2f}%)".format(correct_preds, len(y_test), 100 * correct_preds / len(y_test)))
