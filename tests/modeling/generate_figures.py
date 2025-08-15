import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

from matplotlib.gridspec import GridSpec

parameters = str(sys.argv[1])

sps_path = pyf.catch_parameter(parameters,"SPS")
rps_path = pyf.catch_parameter(parameters,"RPS")
xps_path = pyf.catch_parameter(parameters,"XPS")

SPS = np.loadtxt(sps_path, delimiter = ",", dtype = float)
RPS = np.loadtxt(rps_path, delimiter = ",", dtype = float)
XPS = np.loadtxt(xps_path, delimiter = ",", dtype = int)

nr = len(RPS)

nx = int(pyf.catch_parameter(parameters, "x_samples"))
ny = int(pyf.catch_parameter(parameters, "y_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dh = float(pyf.catch_parameter(parameters, "model_spacing"))

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

model_file = pyf.catch_parameter(parameters, "model_file")

model = pyf.read_binary_volume(nz, nx, ny, model_file)

dh = np.array([dh, dh, dh])
slices = np.array([0.75*nz, 0.75*ny, 0.75*nx], dtype = int)

pyf.plot_model_3D(model, dh, slices, shots = sps_path, scale = 2.0, 
                  nodes = rps_path, adjx = 0.75, dbar = 1.4, cmap = "jet",
                  cblab = "P wave velocity [km/s]")
plt.show()