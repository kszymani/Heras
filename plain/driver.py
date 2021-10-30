from plain.network import Network
from plain.layers import Dense, Activation
from plain.activations import *
from plain.losses import *
from plain.datasets import get_mnist_data


x_train, y_train, x_test, y_test, input_size, test_values = get_mnist_data(seed=9876)
network = Network(seed=5678)

network.add(Dense(input_size, 10))
network.add(Activation(square, square_deriv))
network.add(Dense(10, 1))
network.add(Activation(sigmoid, sigmoid_deriv))

network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

network.fit(x_train, y_train, epochs=10, lr=0.1)
pred = network.predict(x_test)

correct_preds = 0
for e in zip(pred.flatten(), y_test.flatten(), test_values.flatten()):
    p = np.round(e[0])
    correct = p == e[1]
    if correct:
        correct_preds += 1
    print(f"{e[2]} is {'even' if p == 0 else 'odd'} ({'GOOD' if correct else 'BAD'}) ({np.round(e[0], 4)})")

print("Correct predictions: {}/{} ({:.2f}%)".format(correct_preds, len(y_test), 100 * correct_preds / len(y_test)))
