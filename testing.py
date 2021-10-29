from generate_context import restore_from
import numpy as np
from array_utils import encrypt_array, decrypt_array
from activations import relu_deriv, relu
from maths import sqrt, reciprocal, inverse_root
from losses import binary_crossentropy

HE = restore_from("default")



fun_plain = lambda x: np.reciprocal(np.sqrt(x))
fun_enc = inverse_root

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


def binary_ce(p, y):
    return (p - y) / (p - p ** 2)

