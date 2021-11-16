import os

from encrypted.generate_context import restore_HE_from
import numpy as np
from encrypted.array_utils import encrypt_array, decrypt_array
from encrypted.activations import *
from encrypted.maths import sqrt, reciprocal, inverse_root, sigmoid_test, reciprocal_large

HE = restore_HE_from("../keys/light")

fun_plain = lambda x: 1/(1+np.exp(-x))
fun_enc = sigmoid_test
fun_enc2 = sigmoid

start = -3
end = 3
exact = np.array([np.random.rand(20) * (end - start) + start])

res = fun_plain(exact)
print("encrypting array")
enc = encrypt_array(exact, HE)
print("running function")
enc_res = fun_enc(enc, HE)
enc_res2 = fun_enc2(enc, HE)
print("decrypting function")
dec_res = decrypt_array(enc_res, HE)
dec_res2 = decrypt_array(enc_res2, HE)


for e in zip(exact.flatten(), dec_res.flatten(), res.flatten(), dec_res2.flatten()):
    print(f"{fun_enc.__name__}(%.6f) = [enc: %.6f, exact: %.6f]" % (e[0], e[1], e[2]), "taylor approx: {:.6f}".format(e[3]))
