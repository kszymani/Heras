# Heras
Keras-like library for neural network training on homomorphically encrypted data.

## Contribution
The project implements backpropagation algorithm from scratch for arbitrary multilayer perceptron architecture in order to utilize private homomorphic operations. Since the encryption scheme only allows for addition and multiplication, approximations of activation & loss functions and their derivatives are implemented with the use of Taylor's polynomial and Newton's rootfinding method. 

## Known limitations
* Very slow execution time due to the nature of homomorphic scheme, combined with inherent calculation load of neural network training.
* Very high error accumulation due to noise contained in ciphertext, further amplified by approximation errors.
* Need for ciphertext reencryption, thus contradicting the idea of isolated remote training.
