import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

parameters = "../tests/migration/parameters.txt"

nx = 101
ny = 101
nz = 101

dh = np.array([10,10,10])

slices = np.array([0.8*nz, 0.5*ny, 0.5*nx], dtype = int)

sps_path = pyf.catch_parameter(parameters,"SPS")
rps_path = pyf.catch_parameter(parameters,"RPS")
xps_path = pyf.catch_parameter(parameters,"XPS")

model = pyf.read_binary_volume(nz,nx,ny,"../inputs/models/migration_test_vp.bin")
seismic = pyf.read_binary_volume(nz,nx,ny,"../outputs/seismic/RTM_section_101x101x101.bin")

scale = 0.05*np.std(seismic)

sps_path = pyf.catch_parameter(parameters,"SPS")
rps_path = pyf.catch_parameter(parameters,"RPS")

pyf.plot_model_3D(model, dh, slices, scale = 0.3, adjx = 0.6, dbar = 1.4, cmap = "jet",
                  shots = sps_path, nodes = rps_path, cblab = "P wave velocity [m/s]")
plt.show()

pyf.plot_model_3D(seismic, dh, slices, scale = 0.3, adjx = 0.6, dbar = 1.4, cmap = "Greys",
                  shots = sps_path, nodes = rps_path, cblab = "Amplitude", vmin =-scale, vmax = scale)
plt.show()

