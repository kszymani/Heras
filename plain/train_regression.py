from datasets import *
from plain.network import Network
from plain.layers import Dense, Activation, SoftmaxLayer
from plain.activations import *
from plain.losses import *
from random import shuffle


func = lambda x: x ** 2
args = [[[v]] for v in np.linspace(-1, 1, num=200)]
shuffle(args)
x_train = np.array(args)
y_train = func(x_train)

test_args = [[[v]] for v in np.linspace(-1, 1, num=50)]
shuffle(test_args)
x_test = np.array(test_args)
y_test = func(x_test)
network = Network()

network.add(Dense(1, 10))
network.add(Activation(sigmoid, sigmoid_deriv))
network.add(Dense(10, 5))
network.add(Activation(sigmoid, sigmoid_deriv))
network.add(Dense(5, 1))
# network.add(Activation(sigmoid, sigmoid_deriv))
network.set_loss(mse, mse_deriv)

network.fit(x_train, y_train, epochs=15, lr=0.01)
pred = network.predict(x_test)
correct_preds = 0
for e in zip(pred.flatten(), y_test.flatten(),):
    print(e[0])
    rounding = 4
    p = np.round(e[0], rounding)
    y = np.round(e[1], rounding)

    correct = p == y
    if correct:
        correct_preds += 1
    print("predicted {}, actual: {}".format(p, y))
print("Correct predictions: {}/{} ({:.2f}%)".format(correct_preds, len(y_test), 100 * correct_preds / len(y_test)))