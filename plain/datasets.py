from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import random
import numpy as np


def get_mnist_data_categorical(split_ratio=0.25, seed=None, show=False):
    if seed is not None:
        random.seed(seed)
    mnist = load_digits()
    num_img = len(mnist.images)
    num_classes = 10
    if show:
        fig, axes = plt.subplots(2, 10, figsize=(16, 6))
        for i in range(20):
            axes[i // 10, i % 10].imshow(mnist.images[i], cmap='gray')
            axes[i // 10, i % 10].axis('off')
            axes[i // 10, i % 10].set_title(f"target: {mnist.target[i]}")
        plt.tight_layout()
        plt.show()
    print(f"Loaded {num_img} images")
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    test_values = []
    for i in range(num_img):
        x = []
        for r in mnist.images[i]:
            for c in r:
                x.append(c / 255)
        y = [0.0 if j != mnist.target[i] else 1.0 for j in range(num_classes)]
        print(y)
        if random.random() < split_ratio:
            x_test.append([x])
            y_test.append([y])
            test_values.append([mnist.target[i]])
        else:
            x_train.append([x])
            y_train.append([y])
    input_size = len(x_train[0][0])
    print("Train images: ", len(x_train), "Test images: ", len(x_test), "Split ratio: ", split_ratio)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), input_size, np.array(test_values)


def get_mnist_data(split_ratio=0.25, seed=None, show=False, divide_size_by=1):
    if seed is not None:
        random.seed(seed)
    mnist = load_digits()
    if show:
        fig, axes = plt.subplots(2, 10, figsize=(16, 6))
        for i in range(20):
            axes[i // 10, i % 10].imshow(mnist.images[i], cmap='gray')
            axes[i // 10, i % 10].axis('off')
            axes[i // 10, i % 10].set_title(f"target: {mnist.target[i]}")
        plt.tight_layout()
        plt.show()
    num_img = len(mnist.images)
    print(f"Loaded {num_img} images")
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    test_values = []
    for i in range(num_img // divide_size_by):
        x = []
        for r in mnist.images[i]:
            for c in r:
                x.append(c / 255)
        y = [mnist.target[i] % 2]
        if random.random() < split_ratio:
            x_test.append([x])
            y_test.append([y])
            test_values.append([mnist.target[i]])
        else:
            x_train.append([x])
            y_train.append([y])
    input_size = len(x_train[0][0])
    print("Train images: ", len(x_train), "Test images: ", len(x_test), "Split ratio: ", split_ratio)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), input_size, np.array(test_values)


if __name__ == '__main__':
    get_mnist_data(show=True)
