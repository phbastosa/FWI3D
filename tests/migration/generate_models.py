import numpy as np
import matplotlib.pyplot as plt

x_max = 1e3
y_max = 1e3
z_max = 1e3

dh = 10.0

nx = int((x_max / dh) + 1)
ny = int((y_max / dh) + 1)
nz = int((z_max / dh) + 1)

Vp = np.zeros((nz,nx,ny)) + 1500

hz = int(0.8*nz)

Vp[hz-5:hz+5,:,:] += 500

Vp.flatten("F").astype(np.float32, order = "F").tofile(f"../inputs/models/migration_test_vp.bin")
