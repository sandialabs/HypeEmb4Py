import numpy as np
import networkx as nx 
import time as T 
import h5py
from igraph import Graph
import os, sys
from memory_profiler import profile
from tqdm import tqdm

'''
Load graph and compute distances from landmarks to all nodes
'''
# graph_driver = 'networkx'
graph_driver = 'igraph'
print("Using graph driver: ", graph_driver)

network = sys.argv[1]
print("Network: ", network)

HOME = os.getenv("HOME")
datadir = HOME + "/Research/data/"
# clean up
try:
    os.remove(datadir + "D_" + network + ".hdf5")
except:
    print("No clean up necessary.")

nland = int(sys.argv[2])
datatype = 'i4' # 4 byte (32 bit integer)

 # if graph_driver == 'igraph':
# load from igraph
print("Loading data...")
start = T.time()
edgelist = datadir + "com-" + network + ".ungraph.remapped.txt"
# G2 = Graph.Read(edgelist,format="edgelist", directed=False)
G2 = Graph.Read_Edgelist(edgelist, directed=False)
# G2 = Graph.Load(datadir + "igraph_graph_" + network,format="graphmlz")
print("Done Loading! %.3f" %(T.time()-start))
# G2 = Graph.Load("igraph_edgelist",format="edgelist")
# G2 = Graph.as_undirected(G2) # avoid inf when calculating distances
# nodes = G2.vs.indices
nnodes = G2.vcount() #len(nodes)
# print("Nodes ids are in order = ", np.amax(nodes) + 1 == G2.vcount())

# get node degrees for each index
print("Get degrees for each node...")
degrees = G2.vs.degree()
print("Creating node degree map (after remapping)...")
n2d = np.zeros((nnodes,1),dtype=datatype)
# n2d[:,0] = nodes
n2d[:,0] = degrees
# node_degree_map = dict(zip(nodes,degrees))
print("sorting nodes by degree...")
sorted_deg_indices = np.flip(n2d[:,0].argsort())
# ndm_sorted_by_deg = {k: v for k,v in sorted(node_degree_map.items(), key=lambda x: x[1], reverse=True)}

# randomly choose landmark ids
# nland = 5
print("Choosing landmark ids by degree...")
# need to sort for faster indexing
landmark_ids = np.sort(sorted_deg_indices[:nland]) #list(range(nnodes))
landmark_deg = n2d[landmark_ids]
# print("landmark degrees:", landmark_deg)
# np.random.shuffle(landmark_ids)
# landmark_ids = np.sort(landmark_ids[:nland])
# print("Create landmark id map...")
# landmark_map = dict(zip(list(range(nland)),landmark_ids))

import collections

print("Filling in data matrix...")
start = T.time()
f2 = h5py.File(datadir + "D_" + network + ".hdf5", "a",libver='latest')
# # clear data if already exists
# if 'D' in list(f2.keys()):
#     del f2['D'] 
d0 = Graph.shortest_paths(G2,landmark_ids[0])[0]
d0 = np.array(d0,dtype=datatype)
print("creating hdf5 dataset...")
D = f2.create_dataset('D', (1,nnodes), data=d0, 
            maxshape=(nland,nnodes),dtype=datatype,
            compression="gzip",chunks=(nland,10000))
# D_np = np.zeros((nland,nnodes),dtype=datatype)
# D_np[0] = d0
print("starting iterations...")
for i,n in tqdm(enumerate(landmark_ids[1:])):
    # start2 = T.time()
    D.resize(D.shape[0] + 1, axis=0)  
    d = Graph.shortest_paths(G2,n)[0]
    D[-1] = np.array(d,dtype=datatype)
    # if (i+2) % 5 == 0: 
    #     print("Iteration ", i+2,", shape = ", D.shape)
    # D_np[i+1] = list(d)
    # print(i)
# np.save("D_np.npy",D_np)

print("Total time to fill in D is %.3f sec" %(T.time()-start))

# save landmark ids
f2.create_dataset('landmark_ids', (nland,), data=landmark_ids, dtype=datatype)

# save landmark ids
f2.create_dataset('landmark_degrees', (nland,), data=landmark_deg, dtype=datatype)

# save number of edges
f2.attrs["network"] = network
f2.attrs["nedges"] = G2.ecount()
f2.attrs["nnodes"] = G2.vcount()

# close
f2.close()

# # networkx
# @profile
# def test():
#     if graph_driver == 'networkx':
#         # read saved graph after running preproc.py
#         print("Reading graph...")
#         start = T.time()
#         saved_edges = "networkx_edgelist"
#         G = nx.read_edgelist(datadir + saved_edges,nodetype=int)
#         nnodes = len(list(G.nodes()))
#         print("Total time is %.5f sec" %(T.time()-start))

#         # choose landmark_ids
#         # nland = 5
#         landmark_ids = list(range(nnodes))
#         # np.random.shuffle(landmark_ids)
#         landmark_ids = np.sort(landmark_ids[:nland])

#         import collections

#         print("Filling in data matrix...")
#         f = h5py.File(datadir + "D.hdf5", "a",libver='latest')
#         # clear data if already exists
#         if 'D' in list(f.keys()):
#             del f['D']
#         d0 = nx.single_source_shortest_path_length(G,source=landmark_ids[0])
#         od = collections.OrderedDict(sorted(d0.items()))
#         D = f.create_dataset('D', (1,nnodes), data=list(od.values()), 
#                     maxshape=(nland,nnodes),dtype=datatype,
#                     compression="gzip",chunks=(nland,1000)) 
#         # D_np = np.zeros((nland,nnodes),dtype=datatype)
#         # D_np[0] = d0
#         for i,n in enumerate(landmark_ids[1:]):
#             # start2 = T.time()
#             D.resize(D.shape[0] + 1, axis=0)  
#             d = nx.single_source_shortest_path_length(G,source=n)
#             od = collections.OrderedDict(sorted(d.items()))
#             # od = collections.OrderedDict(sorted(d.items()))
#             D[-1] = np.array(list(od.values()),dtype=datatype)
#             if (i+2) % 5 == 0: 
#                 print("Iteration ", i+2,", shape = ", D.shape)
#             # D_np[i+1] = list(d)
#         # np.save("D_np.npy",D_np)

#         print("Total time to fill in D is %.5f sec" %(T.time()-start))
        
#         # close
#         f.close()

   

# if __name__ == '__main__':
#     test()