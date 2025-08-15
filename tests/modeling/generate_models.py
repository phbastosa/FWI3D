import numpy as np

x_max = 6e3
y_max = 4e3
z_max = 2e3

dh = 25.0

nx = int((x_max / dh) + 1)
ny = int((y_max / dh) + 1)
nz = int((z_max / dh) + 1)

Vp = np.zeros((nz, nx, ny)) + 1500

v = np.array([1500, 1700, 1900, 2300, 3000, 3500])
z = np.array([250, 250, 250, 250, 500])

for i in range(len(z)):
    Vp[int(np.sum(z[:i+1]/dh)):] = v[i+1]

Vp[50:70,150:200,100:140] += 1000

Vp.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/modeling_test_vp.bin")
