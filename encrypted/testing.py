import os

from encrypted.generate_context import restore_HE_from
import numpy as np
from array_utils import encrypt_array, decrypt_array
from activations import relu_deriv, relu, sigmoid, sigmoid_deriv
from maths import sqrt, reciprocal, inverse_root

HE = restore_HE_from("keypack")

fun_plain = lambda x: np.reciprocal(x)
fun_enc = reciprocal

np.random.seed(543)
exact = np.random.rand(15)

res = fun_plain(exact)
print("encrypting array")
enc = encrypt_array(exact, HE)
print("running function")
enc_res = fun_enc(enc, HE)
print("decrypting function")
dec_res = decrypt_array(enc_res, HE)

for e in zip(exact, dec_res, res):
    print(f"{fun_enc.__name__.replace('_array', '')}(%.6f) = [enc: %.6f, exact: %.6f]" % (e[0], e[1], e[2]))

