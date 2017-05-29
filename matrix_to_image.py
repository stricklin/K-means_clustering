#!/usr/bin/python2
import netpbm
import os
import numpy as np

load_directory = "superimposed_data/"
save_directory = "superimposed_images/"
files = os.listdir(load_directory)
for file in files:
    data = np.load(load_directory + file)
    netpbm.imsave(save_directory + file + ".pgm", data)
