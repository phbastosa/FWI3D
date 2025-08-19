import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

from scipy import ndimage

parameters = "../tests/migration/parameters.txt"

nx = 101
ny = 101
nz = 51

dh = np.array([10,10,10])

slices = np.array([0.25*nz, 0.5*ny, 0.5*nx], dtype = int)

seismic = pyf.read_binary_volume(nz,nx,ny,"../outputs/seismic/RTM_section_51x101x101.bin")

seismic = ndimage.gaussian_laplace(seismic, sigma=1)

scale = 0.5*np.std(seismic)

sps_path = pyf.catch_parameter(parameters,"SPS")
rps_path = pyf.catch_parameter(parameters,"RPS")

pyf.plot_model_3D(seismic, dh, slices, scale = 0.6, adjx = 0.6, dbar = 1.3, cmap = "jet",
                  cblab = "Amplitude", vmin =-scale, vmax = scale)

plt.show()