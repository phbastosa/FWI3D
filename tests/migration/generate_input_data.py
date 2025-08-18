import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

parameters = str(sys.argv[1])

SPS = np.loadtxt(pyf.catch_parameter(parameters, "SPS"), delimiter = ",", dtype = np.float32) 
RPS = np.loadtxt(pyf.catch_parameter(parameters, "RPS"), delimiter = ",", dtype = np.float32) 
XPS = np.loadtxt(pyf.catch_parameter(parameters, "XPS"), delimiter = ",", dtype = np.int32) 

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

ns = len(SPS)
nr = len(RPS)

input_folder = pyf.catch_parameter(parameters, "mod_output_folder")
output_folder = pyf.catch_parameter(parameters, "mig_input_folder")

for sId in range(ns):

    data_file = f"seismogram_nt{nt}_nr{nr}_{int(1e6*dt)}us_shot_{sId+1}.bin"

    seismic = pyf.read_binary_matrix(nt, nr, input_folder + data_file)

    travel_time = np.sqrt((SPS[sId,0] - RPS[:,0])**2 + 
                          (SPS[sId,1] - RPS[:,1])**2 + 
                          (SPS[sId,2] - RPS[:,2])**2) / 1500

    tId = np.array((travel_time + 0.15) / dt, dtype = int)

    for rId in range(nr):
        seismic[:tId[rId], rId] = 0.0

    seismic.flatten("F").astype(np.float32, order = "F").tofile(output_folder + data_file)
