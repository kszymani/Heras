from encrypted.generate_context import restore_HE_from
from encrypted.array_utils import encrypt_array, decrypt_array
from encrypted.network import Network
from encrypted.layers import Dense, Activation
from encrypted.activations import *
from encrypted.losses import *
from datasets import get_arrows

HE = restore_HE_from("../keys/light")

x_train, y_train, x_test, y_test, input_size = get_arrows(size=(12, 12), show=False)
x_train = x_train[:400]
y_train = y_train[:400]

print("Initializing network")
network = Network(seed=248263)
network.add(Dense(input_size, 10, HE))
network.add(Activation(relu_client, relu_client_deriv))
network.add(Dense(10, 5, HE))
network.add(Activation(relu_client, relu_client_deriv))
network.add(Dense(5, 1, HE))
network.add(Activation(sigmoid, sigmoid_deriv))
network.set_loss(binary_crossentropy, binary_crossentropy_deriv)

epochs = 1
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

network.save_weights("arrow_weights", HE)

test_size = len(x_test)

correct_preds = 0
preds = 0
for i in range(test_size):
    print(f"Predicting sample {i + 1}/{test_size}")
    x_enc = encrypt_array(x_test[i], HE)
    y = y_test[i][0]
    pred = decrypt_array(network.predict_sample(x_enc, HE), HE)[0, 0]

    p = np.round(pred)
    correct = p == y
    if correct:
        correct_preds += 1
    preds += 1
    print(f" Arrow is pointing {'LEFT' if p == 0 else 'RIGHT'}({'GOOD' if correct else 'BAD'}) ({np.round(pred, 4)})")
print("Correct predictions: {}/{} ({:.2f}%)".format(correct_preds, preds, 100 * correct_preds / preds))
