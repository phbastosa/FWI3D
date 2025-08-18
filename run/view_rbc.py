import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

parameters = "../tests/migration/parameters.txt"

nx = 203
ny = 203
nz = 153

dh = np.array([10,10,10])

slices = np.array([0.5*nz, 0.5*ny, 0.5*nx], dtype = int)

model = pyf.read_binary_volume(nz, nx, ny, "vp_expanded.bin")

sps_path = pyf.catch_parameter(parameters,"SPS")
rps_path = pyf.catch_parameter(parameters,"RPS")

pyf.plot_model_3D(model, dh, slices, scale = 0.6, adjx = 0.6, dbar = 1.3, cmap = "jet",
                  cblab = "P wave velocity [km/s]")

plt.show()