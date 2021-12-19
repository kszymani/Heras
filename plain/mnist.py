"""
Kod generujący jawną sieć neuronową i przeprowadzający jej trening i test.
"""
from datetime import datetime
from datasets import *
from plain.network import Network
from plain.layers import Dense, Activation
from plain.activations import *
from plain.losses import *

seed = 6079
x_train, y_train, x_test, y_test, input_size, test_values = get_mnist_data_binary(seed=seed)

print("seed: ", seed)
network = Network(seed=seed)

network.add(Dense(input_size, 10))
network.add(Activation(polynomial_plain, polynomial_deriv_plain))
network.add(Dense(10, 1))
network.add(Activation(sigmoid_plain, sigmoid_deriv_plain))
network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

start_time = datetime.now()
network.fit(x_train, y_train, epochs=1, lr=0.01)
end_time = datetime.now()

pred = network.predict(x_test)
correct_preds = 0
for e in zip(pred, y_test, test_values):
    p = np.round(e[0])[0, 0]
    y = e[1][0, 0]
    value = e[2][0]

    correct = p == y
    if correct:
        correct_preds += 1
    print(f"{value} is {'even' if p == 0 else 'odd'} ({'GOOD' if correct else 'BAD'}) ({np.round(e[0][0,0], 4)})")

print("Correct predictions: {}/{} ({:.2f}%)".format(correct_preds, len(y_test), 100 * correct_preds / len(y_test)))
print('Duration: {}'.format(end_time - start_time))
