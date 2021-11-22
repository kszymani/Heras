import os

from encrypted.generate_context import restore_HE_from
import numpy as np
from encrypted.array_utils import encrypt_array, decrypt_array
from encrypted.activations import *
from encrypted.maths import *



HE = restore_HE_from("../keys/light2")

fun_plain = lambda x: 1/x
fun_enc = reciprocal_taylor
fun2 = reciprocal

start = 0
end = 0.5
exact = np.array([np.random.rand(20) * (end - start) + start])

res = fun_plain(exact)
print("encrypting array")
enc = encrypt_array(exact, HE)
print("running function")

enc_res = fun_enc(enc, HE)
enc_res2 = fun2(enc, HE)
print("decrypting function")
dec_res = decrypt_array(enc_res, HE)
dec2 = decrypt_array(enc_res2, HE)

for e in zip(exact.flatten(), dec_res.flatten(), res.flatten(), dec2.flatten()):
    print(f"{fun_enc.__name__}(%.6f) = [enc: %.6f, ,\tenc2: %.6f, \texact: %.6f]" % (e[0], e[1], e[3],e[2]), )
