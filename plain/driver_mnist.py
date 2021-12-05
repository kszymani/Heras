from datasets import *
from plain.network import Network
from plain.layers import Dense, Activation, SoftmaxLayer
from plain.activations import *
from plain.losses import *

seed = random.randint(0, 10000)
x_train, y_train, x_test, y_test, input_size, test_values = get_mnist_data_categorical(seed=seed)
# x_train, y_train, x_test, y_test, input_size, test_values = get_mnist_data(seed=seed)

print("seed: ", seed)
network = Network(seed=seed)

network.add(Dense(input_size, 5))
network.add(Activation(polynomial_plain, polynomial_deriv_plain))
network.add(Dense(5, 10))
network.add(Activation(sigmoid_plain, sigmoid_deriv_plain))
# network.add(Activation(sigmoid_scaled_plain, sigmoid_scaled_deriv_plain))
network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

network.fit(x_train, y_train, epochs=3, lr=0.01)
pred = network.predict(x_test)
correct_preds = 0
for e in zip(pred, y_test, test_values):
    p = np.argmax(e[0][0])
    probablity = e[0][0][p]
    y = e[2][0]
    # p = np.round(e[0])[0,0]
    # y = e[1][0,0]
    # value = e[2][0]
    correct = p == y
    if correct:
        correct_preds += 1
    # print(f"{value} is {'even' if p == 0 else 'odd'} ({'GOOD' if correct else 'BAD'}) ({np.round(e[0][0,0], 4)})")
    # print("predicted {}, actual: {} ({:.6f})".format(p, y, probablity))

print("Correct predictions: {}/{} ({:.2f}%)".format(correct_preds, len(y_test), 100 * correct_preds / len(y_test)))
print(seed)