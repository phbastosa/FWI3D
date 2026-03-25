import sys; sys.path.append("../src/")

import numpy as np
import functions as pyf

from scipy.ndimage import gaussian_filter

parameters = str(sys.argv[1])

SPS_path = pyf.catch_parameter(parameters,"SPS") 
RPS_path = pyf.catch_parameter(parameters,"RPS") 

model_file = pyf.catch_parameter(parameters,"model_file")

nx = int(pyf.catch_parameter(parameters, "x_samples"))
ny = int(pyf.catch_parameter(parameters, "y_samples"))
nz = int(pyf.catch_parameter(parameters, "z_samples")) 

dh = float(pyf.catch_parameter(parameters, "model_spacing"))

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

nsx = 6
nsy = 6

nrx = 51
nry = 51

ns = nsx*nsy
nr = nrx*nry

SPS = np.zeros((ns, 3))
RPS = np.zeros((nr, 3))

sx, sy = np.meshgrid(np.linspace(1000, 4000, nsx), 
                     np.linspace(1000, 4000, nsy))

rx, ry = np.meshgrid(np.linspace(0, 5000, nrx), 
                     np.linspace(0, 5000, nry))

SPS[:,0] = np.reshape(sx, [ns], order = "F")
SPS[:,1] = np.reshape(sy, [ns], order = "F")
SPS[:,2] = np.zeros(ns) + 100.0 

RPS[:,0] = np.reshape(rx, [nr], order = "F")
RPS[:,1] = np.reshape(ry, [nr], order = "F")
RPS[:,2] = np.zeros(nr) 

np.savetxt(SPS_path, SPS, fmt = "%.2f", delimiter = ",")
np.savetxt(RPS_path, RPS, fmt = "%.2f", delimiter = ",")

m_true = np.fromfile(model_file, dtype = np.float32, count = nx*ny*nz).reshape([nz,nx,ny], order = "F")

vmin = np.min(m_true)
vmax = np.max(m_true)

m_init = m_true.copy()

zmask = float(pyf.catch_parameter(parameters, "depth_mask"))

zId = int(zmask/dh)

m_init = 1.0 / gaussian_filter(1.0 / m_true, 3.0)

m_init[:zId] = m_true[:zId]

m_init.flatten("F").astype(np.float32, order = "F").tofile(model_file.replace("true", "init"))