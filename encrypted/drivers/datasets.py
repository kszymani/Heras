from sklearn.datasets import load_digits
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
import numpy as np


def get_arrows(size=(12, 12), show=True):
    train_labels = np.load('arrows/labels_arr_train.npz')
    train_labels = train_labels.f.arr_0
    train_labels = np.expand_dims(train_labels, axis=-1)

    test_labels = np.load('arrows/labels_arr_train.npz')
    test_labels = test_labels.f.arr_0
    test_labels = np.expand_dims(test_labels, axis=-1)

    train_img = np.load('arrows/aug_arr_train.npz')
    train_img = train_img.f.arr_0
    resized_train = []
    for i, img in enumerate(train_img):
        resized_train.append(resize(img, size, anti_aliasing=True))
    print("Loaded ", len(resized_train), " train images")
    print("Image size: ", len(resized_train[0]))

    test_img = np.load('arrows/aug_arr_test.npz')
    test_img = test_img.f.arr_0
    resized_test = []
    for i, img in enumerate(test_img):
        resized_test.append(resize(img, size, anti_aliasing=True))
    print("Loaded ", len(resized_test), " test images")
    print("Image size: ", len(resized_test[0]))

    if show:
        fig, axes = plt.subplots(2, 10, figsize=(16, 6))
        for i in range(20):
            axes[i // 10, i % 10].imshow(resized_train[i], cmap='gray')
            axes[i // 10, i % 10].axis('off')
            axes[i // 10, i % 10].set_title(f"target: {'left' if train_labels[i][0] == 0 else 'right'}")
        plt.tight_layout()
        plt.show()
    resized_train = [np.expand_dims(img.flatten(), axis=0) for img in resized_train]
    resized_test = [np.expand_dims(img.flatten(), axis=0) for img in resized_test]
    return np.array(resized_train), train_labels, np.array(resized_test), test_labels, size[0] * size[1]


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
    get_arrows()
    # get_mnist_data(show=True)
