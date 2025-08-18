import numpy as np
import matplotlib.pyplot as plt

x_max = 1e3
y_max = 1e3
z_max = 5e2

dh = 10.0

nx = int((x_max / dh) + 1)
ny = int((y_max / dh) + 1)
nz = int((z_max / dh) + 1)

Vp = np.zeros((nz,nx,ny)) + 1500

hx = int(0.50*nx)
hy = int(0.50*ny)
hz = int(0.75*nz)

Vp[hz-1:hz+2,hx-1:hx+2,hy-1:hy+2] += 500

Vp.flatten("F").astype(np.float32, order = "F").tofile(f"../inputs/models/migration_test_vp.bin")
