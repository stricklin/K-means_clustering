#!/usr/bin/python2
import netpbm
import os
import numpy as np

load_directory = "30_centroids/"
save_directory = "30_centroid_images/"
files = os.listdir(load_directory)
for file in files:
    data = np.load(load_directory + file).astype(np.int64)
    netpbm.imsave(save_directory + file + ".pgm", data)
