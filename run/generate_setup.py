import sys; sys.path.append("../src/")

import numpy as np
import functions as pyf
import matplotlib.pyplot as plt

parameters = str(sys.argv[1])

SPS_path = pyf.catch_parameter(parameters,"SPS") 
RPS_path = pyf.catch_parameter(parameters,"RPS") 
XPS_path = pyf.catch_parameter(parameters,"XPS")

model_file = pyf.catch_parameter(parameters,"model_file")

nx = int(pyf.catch_parameter(parameters, "x_samples"))
ny = int(pyf.catch_parameter(parameters, "y_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dh = float(pyf.catch_parameter(parameters, "model_spacing"))

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

vp = np.array([1500, 1800, 2000])
z = np.array([750, 1000])

nsx = 11
nsy = 11

nrx = 201
nry = 201

ns = nsx*nsy
nr = nrx*nry

SPS = np.zeros((ns, 3))
RPS = np.zeros((nr, 3))
XPS = np.zeros((ns, 3))

sx, sy = np.meshgrid(np.linspace(500, 4500, nsx), 
                     np.linspace(500, 4500, nsy))

rx, ry = np.meshgrid(np.linspace(0, 5000, nrx), 
                     np.linspace(0, 5000, nry))

SPS[:,0] = np.reshape(sx, [ns], order = "F")
SPS[:,1] = np.reshape(sy, [ns], order = "F")
SPS[:,2] = np.zeros(ns) + 100.0 

RPS[:,0] = np.reshape(rx, [nr], order = "F")
RPS[:,1] = np.reshape(ry, [nr], order = "F")
RPS[:,2] = np.zeros(nr) 

XPS[:, 0] = np.arange(ns)
XPS[:, 1] = np.zeros(ns) 
XPS[:, 2] = np.zeros(ns) + nr 

np.savetxt(SPS_path, SPS, fmt = "%.2f", delimiter = ",")
np.savetxt(RPS_path, RPS, fmt = "%.2f", delimiter = ",")
np.savetxt(XPS_path, XPS, fmt = "%.0f", delimiter = ",")

vp = np.array([1500, 1800, 2000])
z = np.array([750, 1000])

Vp = np.zeros((nz,nx,ny))

for i in range(len(vp)):
    layer = int(np.sum(z[:i])/dh)
    Vp[layer:] = vp[i]

Vp.flatten("F").astype(np.float32, order = "F").tofile(model_file)

dh = np.array([dh, dh, dh])
slices = np.array([0.1*nz, 0.5*ny, 0.5*nx], dtype = int)

pyf.plot_model_3D(Vp, dh, slices, shots = SPS_path, scale = 1.4, 
                  nodes = RPS_path, adjx = 0.7, dbar = 1.4, cmap = "jet",
                  cblab = "P wave velocity [km/s]")
plt.show()
