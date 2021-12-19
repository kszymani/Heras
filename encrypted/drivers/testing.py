from datetime import datetime

from encrypted.generate_context import restore_HE_from
from encrypted.maths import *
from plain.activations import *

HE = restore_HE_from("../keys/light")

# fun_plain = sigmoid_plain

fun_plain = lambda x: np.log(x)
fun_enc = log

start = 0.1
stop = 1

num = 20
# exact = np.array([np.random.rand(200) * (stop - start) + start])
exact = np.linspace(start, stop, num=num,)

res = fun_plain(exact)

print("encrypting array")
enc = encrypt_array(exact, HE)

print("running function")
start_time = datetime.now()
enc_res = fun_enc(enc, HE)
# enc_res = sigmoid_extended(enc, HE, coeffs_map=get_map_sigmoid(HE))
end_time = datetime.now()

print("decrypting function")

dec_res = decrypt_array(enc_res, HE)

error = 0
for e in zip(exact.flatten(), dec_res.flatten(), res.flatten()):
    err = np.abs((e[2] - e[1])/e[1])
    error += err
    # print(f"{fun_plain.__name__}(%.6f) = [enc: %.6f, \texact: %.6f] rel_error: %.6f" % (e[0], e[1], e[2], err))
    print("%.4f & %.8f & %.8f \\\\ \hline" % (e[0], e[1], e[2],))

t = end_time - start_time
print(fun_enc.__name__)
print('Duration: {}'.format(t))
print('Single input duration: {}'.format(t / num))
print('Mean realtive error: {:.8f}'.format(error/num))

