import sys; sys.path.append("../src/")

import numpy as np
import functions as pyf
import matplotlib.pyplot as plt

parameters = str(sys.argv[1])

sps_path = pyf.catch_parameter(parameters,"SPS") 
rps_path = pyf.catch_parameter(parameters,"RPS") 

nx = int(pyf.catch_parameter(parameters, "x_samples"))
ny = int(pyf.catch_parameter(parameters, "y_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dh = float(pyf.catch_parameter(parameters, "model_spacing"))

image_file = "../outputs/seismic/RTM_section_81x201x201.bin"
model_file = pyf.catch_parameter(parameters,"model_file")

image = pyf.read_binary_volume(nz,nx,ny,image_file)
model = pyf.read_binary_volume(nz,nx,ny,model_file)

SPS = np.loadtxt(sps_path, dtype = np.float32, delimiter = ",")
RPS = np.loadtxt(rps_path, dtype = np.float32, delimiter = ",")

dh = np.array([dh, dh, dh])
slices = np.array([0.05*nz, 0.2*ny, 0.2*nx], dtype = int)

pyf.plot_model_3D(model, dh, slices, shots = sps_path, scale = 1.4, 
                  adjx = 0.7, dbar = 1.4, cmap = "jet",
                  cblab = "P wave velocity [km/s]")
plt.show()

image *= 1000.0 / np.max(np.abs(image))

pyf.plot_model_3D(image, dh, slices, shots = sps_path, scale = 1.4, 
                  adjx = 0.7, dbar = 1.4, cmap = "Greys",
                  vmin = -500, vmax = 500, cblab = "Normalized Amplitude")
plt.show()
