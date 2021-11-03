
from network import Network
from layers import Dense, Activation
from activations import *
from losses import *
from plain.datasets import get_mnist_data


x_train, y_train, x_test, y_test, input_size, test_values = get_mnist_data(seed=9876, divide_size_by=1)
network = Network(seed=5678)

network.add(Dense(input_size, 5))
network.add(Activation(square, square_deriv))
network.add(Dense(5, 1))
network.add(Activation(sigmoid_squared, sigmoid_squared_deriv))

network.set_loss(binary_crossentropy, binary_crossentropy_deriv)
epochs = 2
for j in range(epochs):
    print("\n\nEpoch ", j)
    for i in range(len(x_train)):
        print("Sample {}/{}".format(i, len(x_train)))
        x = x_train[i]
        y = y_train[i]
        err = network.fit_sample(x, y, lr=0.1)
        print("Loss: ", err)


correct_preds = 0
preds = 0
for i in range(len(x_test)):
    x = x_test[i]
    y = y_test[i]
    value = test_values[i][0]
    pred = network.predict_sample(x)[0]
    p = np.round(pred)[0]
    correct = p == y
    if correct:
        correct_preds += 1
    preds += 1
    print(f"{value} is {'even' if p == 0 else 'odd'} ({'GOOD' if correct else 'BAD'}) ({np.round(pred[0], 4)})")

print("Correct predictions: {}/{} ({:.2f}%)".format(correct_preds, preds, 100 * correct_preds / preds))
