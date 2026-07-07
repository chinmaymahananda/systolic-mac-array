import numpy as np
q = np.load("model_int8.npz")
gv = np.load("golden_vectors.npz")
w1 = q["w1"]
x_q = gv["inputs"][0]
co = 0
oy, ox = 0, 0
running = 0
print(f"channel {co}, pixel (oy={oy},ox={ox})")
for ky in range(3):
    for kx in range(3):
        a = int(x_q[oy+ky, ox+kx])
        w = int(w1[co, 0, ky, kx])
        p = a * w
        running += p
        print(f"tap(ky={ky},kx={kx}) img_addr={(oy+ky)*8+(ox+kx):2d} a={a:4d} w={w:4d} product={p:6d} running_sum={running:6d}")
b1 = q["b1"]
print(f"bias[{co}] = {int(b1[co])}")
print(f"final (sum+bias) = {running + int(b1[co])}")
print("golden conv1_acc[0][co][oy][ox] =", gv["conv1_acc"][0][co][oy][ox])

