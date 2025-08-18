import numpy as np

nsx = 3
nsy = 3

nrx = 41 
nry = 41

ns = nsx*nsy
nr = nrx*nry

sx, sy = np.meshgrid(np.linspace(310, 690, nsx), 
                     np.linspace(310, 690, nsy))

rx, ry = np.meshgrid(np.linspace(100, 900, nrx), 
                     np.linspace(100, 900, nry))

SPS = np.zeros((ns, 3), dtype = float)
RPS = np.zeros((nr, 3), dtype = float)
XPS = np.zeros((ns, 3), dtype = int)

SPS[:,0] = np.reshape(sx, [ns], order = "F")
SPS[:,1] = np.reshape(sy, [ns], order = "F")
SPS[:,2] = np.zeros(ns)

RPS[:,0] = np.reshape(rx, [nr], order = "C")
RPS[:,1] = np.reshape(ry, [nr], order = "C")
RPS[:,2] = np.zeros(nr)

XPS[:, 0] = np.arange(ns)
XPS[:, 1] = np.zeros(ns) 
XPS[:, 2] = np.zeros(ns) + nr 

path_SPS = "../inputs/geometry/migration_test_SPS.txt"
path_RPS = "../inputs/geometry/migration_test_RPS.txt"
path_XPS = "../inputs/geometry/migration_test_XPS.txt"

np.savetxt(path_SPS, SPS, fmt = "%.2f", delimiter = ",")
np.savetxt(path_RPS, RPS, fmt = "%.2f", delimiter = ",")
np.savetxt(path_XPS, XPS, fmt = "%.0f", delimiter = ",")
