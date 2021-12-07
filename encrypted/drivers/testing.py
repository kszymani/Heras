import os

from encrypted.generate_context import restore_HE_from
import numpy as np
from encrypted.array_utils import encrypt_array, decrypt_array
from encrypted.activations import *
from encrypted.maths import *
from plain.activations import *
from encrypted.layers import ExtendedActivation

HE = restore_HE_from("../keys/light")


fun_plain = sigmoid_deriv_plain
# fun_enc = sigmoid_

start = -5
end = 5
exact = np.array([np.random.rand(20) * (end - start) + start])

res = fun_plain(exact)
print("encrypting array")
enc = encrypt_array(exact, HE)
print("running function")

# enc_res = fun_enc(enc, HE)

enc_res = sigmoid_extended_deriv(enc, get_map_sigmoid_deriv(HE), HE)
print("decrypting function")
dec_res = decrypt_array(enc_res, HE)

for e in zip(exact.flatten(), dec_res.flatten(), res.flatten()):
    print(f"{fun_plain.__name__}(%.6f) = [enc: %.6f, \texact: %.6f]" % (e[0], e[1], e[2]), )
