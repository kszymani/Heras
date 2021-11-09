import os

from encrypted.generate_context import restore_HE_from
import numpy as np
from encrypted.array_utils import encrypt_array, decrypt_array
from encrypted.activations import *
from encrypted.maths import sqrt, reciprocal, inverse_root

HE = restore_HE_from("light")

fun_plain = lambda x: 2* 1/(1+np.exp(-x)) * 1/(1+np.exp(-x)) * (1 - 1/(1+np.exp(-x)) )
fun_enc = sigmoid_squared_deriv

exact = np.random.rand(30)

res = fun_plain(exact)
print("encrypting array")
enc = encrypt_array(exact, HE)
print("running function")
enc_res = fun_enc(enc, HE)
print("decrypting function")
dec_res = decrypt_array(enc_res, HE)

for e in zip(exact, dec_res, res):
    if e[1] > 1.0 or e[1] < 0.0:
        print("ERROR")
    print(f"{fun_enc.__name__}(%.6f) = [enc: %.6f, exact: %.6f]" % (e[0], e[1], e[2]))


