import numpy as np
import networkx as nx 
import time as T 
import h5py

'''
Read the distance matrix for the particular graph
'''

# read data set
print("Loading data...")
start = T.time()
fread = h5py.File("D.hdf5", "r")
D = fread['D']
# D_np = np.load("D_np.npy")
print("Time %.5f" %(T.time()-start))


