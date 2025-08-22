import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

parameters = str(sys.argv[1])

nx = int(pyf.catch_parameter(parameters, "x_samples"))
ny = int(pyf.catch_parameter(parameters, "y_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dh = float(pyf.catch_parameter(parameters, "model_spacing"))

dh = np.array([dh, dh, dh])

slices = np.array([0.8*nz, 0.5*ny, 0.5*nx], dtype = int)

sps_path = pyf.catch_parameter(parameters,"SPS")
rps_path = pyf.catch_parameter(parameters,"RPS")
xps_path = pyf.catch_parameter(parameters,"XPS")

image_folder = pyf.catch_parameter(parameters, "mig_output_folder")

image = pyf.read_binary_volume(nz,nx,ny, image_folder + f"RTM_section_{nz}x{nx}x{ny}.bin")
model = pyf.read_binary_volume(nz,nx,ny, pyf.catch_parameter(parameters, "model_file"))

image *= 1e5 / np.max(np.abs(image))

scale = np.std(image)

pyf.plot_model_3D(model, dh, slices, scale = 0.4, adjx = 0.5, dbar = 1.25, cmap = "jet",
                  shots = sps_path, nodes = rps_path, cblab = "P wave velocity [m/s]")
plt.show()

pyf.plot_model_3D(image, dh, slices, scale = 0.4, adjx = 0.5, dbar = 1.25, cmap = "Greys",
                  shots = sps_path, nodes = rps_path, cblab = "Amplitude", vmin =-scale, vmax = scale)
plt.show()

