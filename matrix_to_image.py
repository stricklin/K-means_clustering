#!/usr/bin/python2
import netpbm
import os
import numpy as np

load_directory = "centroids/"
save_directory = "centroid_images/"
files = os.listdir(load_directory)
for file in files:
    data = np.load(load_directory + file)
    netpbm.imsave(save_directory + file + ".pgm", data)
