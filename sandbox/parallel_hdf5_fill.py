import numpy as np
from multiprocessing import Pool
import os, sys
import time as T
import h5py
# clean up data file 
try:
    os.remove("test.hdf5")
except:
    print("No hdf5 file to remove!")

# size variables
nland = 100
nnodes = 10**3
nprocs = 2
datatype = 'i4'
rn = np.random.RandomState(234)

# Initialize hdf5 file for input
print("Filling in data matrix...")
f = h5py.File("test.hdf5", "w",libver='latest')
D = f.create_dataset('D', (nland,nnodes), 
            dtype=datatype, compression="gzip")
D_np = np.zeros((nland,nnodes),dtype=datatype)

# split rows up
allrows = np.arange(nland)
rgroups = np.array_split(allrows,nprocs)

# function to generate row (simulate shortest path gen)
def sssp(rows, rstate=rn, maxd=10000):
    d = {}
    for r in rows:
        # d = rn.randint(maxd,size=(1,nnodes))
        d[r] = r*np.ones((1,nnodes))
    return d

def pfill(group):
    rows = rgroups[group]
    d = sssp(rows)
    for k,v in d.items():
        D_np[k] = v

p = Pool(nprocs) 
start = T.time()
res = p.map(sssp,[rows for rows in rgroups])
end = T.time()
p.close()
p.terminate()

print(D[:,:])

# f.close()


# print("starting iterations to fill data...")
# for i,n in tqdm(enumerate(landmark_ids[1:])):
#     # start2 = T.time()
#     D[i] 
#     d = Graph.shortest_paths(G2,n)[0]
#     D[-1] = np.array(d,dtype=datatype)
#     # if (i+2) % 5 == 0: 
#     #     print("Iteration ", i+2,", shape = ", D.shape)
#     # D_np[i+1] = list(d)
#     # print(i)
# # np.save("D_np.npy",D_np)