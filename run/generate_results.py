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

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

SPS = np.loadtxt(sps_path, dtype = np.float32, delimiter = ",")
RPS = np.loadtxt(rps_path, dtype = np.float32, delimiter = ",")

nr = len(RPS)

nrx = int(np.sqrt(nr))
nry = int(np.sqrt(nr))

dr = RPS[1,1] - RPS[0,1]

ns = len(SPS)

# Modeling results

model_true_file = "../inputs/models/overthrust_true_81x201x201_25m.bin"

model_true = pyf.read_binary_volume(nz, nx, ny, model_true_file)

dh = np.array([dh, dh, dh])

slices = np.array([0.75*nz, 0.56*ny, 0.56*nx], dtype = int)

pyf.plot_model_3D(model_true, dh, slices, shots = sps_path, scale = 1.4, 
                  adjx = 0.7, dbar = 1.4, cmap = "jet",
                  cblab = "P wave velocity [km/s]")
plt.savefig("model_true.png", dpi = 200)
plt.show()




sId = 15

dobs = pyf.read_binary_matrix(nt, nr, f"../inputs/data/seismogram_nt{nt}_nr{nr}_{int(dt*1e6)}us_shot_{sId+1}.bin")

fiveShots = slice(int(0.5*nr - 2*nrx), int(0.5*nr + 2*nrx))

scale = 2.0*np.std(dobs)

tloc = np.linspace(0, nt-1, 11)
tlab = np.around(np.linspace(0, (nt-1)*dt, 11), decimals = 1)

traceId = np.arange(nr)

xlab = traceId[fiveShots]

xloc = np.arange(len(xlab))[::25]

xlab = xlab[::25]

fig, ax = plt.subplots(figsize = (6, 8))

ax.imshow(dobs[:,fiveShots], aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

ax.set_xlabel("Trace ID", fontsize = 15)
ax.set_ylabel("Time [s]", fontsize = 15)

ax.set_yticks(tloc)
ax.set_yticklabels(tlab)

ax.set_xticks(xloc)
ax.set_xticklabels(xlab)

fig.tight_layout()
plt.savefig("dobs.png", dpi = 200)
plt.show()

# Migration results

model_init_file = "../inputs/models/overthrust_init_81x201x201_25m.bin"

model_init = pyf.read_binary_volume(nz, nx, ny, model_init_file)

slices = np.array([0.75*nz, 0.56*ny, 0.56*nx], dtype = int)

pyf.plot_model_3D(model_init, dh, slices, shots = sps_path, scale = 1.4, 
                  adjx = 0.7, dbar = 1.4, cmap = "jet",
                  cblab = "P wave velocity [km/s]")
plt.savefig("model_RTM.png", dpi = 200)
plt.show()



dmig = pyf.read_binary_matrix(nt, nr, f"../inputs/data/input_RTM_seismogram_nt{nt}_nr{nr}_{int(dt*1e6)}us_shot_{sId+1}.bin")

scale = 2.0*np.std(dmig)

fig, ax = plt.subplots(figsize = (6, 8))

ax.imshow(dmig[:,fiveShots], aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

ax.set_xlabel("Trace ID", fontsize = 15)
ax.set_ylabel("Time [s]", fontsize = 15)

ax.set_yticks(tloc)
ax.set_yticklabels(tlab)

ax.set_xticks(xloc)
ax.set_xticklabels(xlab)

fig.tight_layout()
plt.savefig("dmig_RTM.png", dpi = 200)
plt.show()



image_file = f"../outputs/seismic/RTM_section_{nz}x{nx}x{ny}.bin"

image = pyf.read_binary_volume(nz, nx, ny, image_file)

image *= 1000 / np.max(image)


slices = np.array([0.75*nz, 0.56*ny, 0.56*nx], dtype = int)

pyf.plot_model_3D(image, dh, slices, shots = sps_path, scale = 1.4, 
                  adjx = 0.7, dbar = 1.4, cmap = "Greys", vmin = -600, 
                  vmax = 600, cblab = "Amplitude Normalized")
plt.savefig("image_RTM.png", dpi = 200)
plt.show()




# Inversion results

convergence_file = "../outputs/residuo/convergence_5_iterations.txt"

convergence = np.loadtxt(convergence_file, dtype = np.float32)

convergence *= 100.0 / np.max(convergence)

fig, ax = plt.subplots(figsize = (10,4))

ax.plot(convergence, "ok--")

ax.set_title("Convergence map", fontsize = 18)
ax.set_xlabel("Iterations", fontsize = 15)
ax.set_ylabel(r"Residuo L$_2$-norm: $|d^{obs} - d^{cal}|^2_2$")

fig.tight_layout()
plt.savefig("residuo_FWI.png", dpi = 200)
plt.show()




model_pred_file = "../outputs/models/model_FWI_50Hz_81x201x201.bin"

model_init = pyf.read_binary_volume(nz, nx, ny, model_init_file)

slices = np.array([0.75*nz, 0.56*ny, 0.56*nx], dtype = int)

pyf.plot_model_3D(model_init, dh, slices, shots = sps_path, scale = 1.4, 
                  adjx = 0.7, dbar = 1.4, cmap = "jet",
                  cblab = "P wave velocity [km/s]")
plt.savefig("model_RTM.png", dpi = 200)
plt.show()
