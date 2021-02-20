import numpy as np
import networkx as nx 
import time as T 
import h5py
from igraph import Graph
import os

'''
Load graph and compute distances from landmarks to all nodes
'''
# graph_driver = 'networkx'
graph_driver = 'igraph'
print("Using graph driver: ", graph_driver)

HOME = os.getenv("HOME")
datadir = HOME + "/Research/data/"

nland = 100
datatype = 'i4' # 4 byte (32 bit integer)

# networkx
if graph_driver == 'networkx':
    # read saved graph after running preproc.py
    print("Reading graph...")
    start = T.time()
    saved_edges = "networkx_edgelist"
    G = nx.read_edgelist(datadir + saved_edges,nodetype=int)
    nnodes = len(list(G.nodes()))
    print("Total time is %.5f sec" %(T.time()-start))

    # choose landmark_ids
    # nland = 5
    landmark_ids = list(range(nnodes))
    # np.random.shuffle(landmark_ids)
    landmark_ids = np.sort(landmark_ids[:nland])

    import collections

    # print("Filling in data matrix...")
    # # start = T.time()
    # # D_np = np.zeros((nland,nnodes),dtype=datatype)
    # for i,n in enumerate(landmark_ids):
    #     if i % 10 == 0: print(i)
    #     start2 = T.time()
    #     d = nx.single_source_shortest_path_length(G,source=n)
    #     od = collections.OrderedDict(sorted(d.items()))
    #     # D[i,list(od.keys())] = list(od.values())
    #     # D_np[i,list(od.keys())] = list(od.values())
    # # np.save("D_np.npy",D_np)
    print("Filling in data matrix...")
    f = h5py.File(datadir + "D.hdf5", "a",libver='latest')
    # clear data if already exists
    if 'D' in list(f.keys()):
        del f['D']
    d0 = nx.single_source_shortest_path_length(G,source=landmark_ids[0])
    od = collections.OrderedDict(sorted(d0.items()))
    D = f.create_dataset('D', (1,nnodes), data=list(od.values()), 
                maxshape=(nland,nnodes),dtype=datatype,
                compression="gzip",chunks=(nland,1000)) 
    # D_np = np.zeros((nland,nnodes),dtype=datatype)
    # D_np[0] = d0
    for i,n in enumerate(landmark_ids[1:]):
        # start2 = T.time()
        D.resize(D.shape[0] + 1, axis=0)  
        d = nx.single_source_shortest_path_length(G,source=n)
        od = collections.OrderedDict(sorted(d.items()))
        # od = collections.OrderedDict(sorted(d.items()))
        D[-1] = np.array(list(d.values()),dtype=datatype)
        if (i+2) % 5 == 0: 
            print("Iteration ", i+2,", shape = ", D.shape)
        # D_np[i+1] = list(d)
    # np.save("D_np.npy",D_np)

    # use h5py to save and retrieve data
    f = h5py.File("D.hdf5", "w", libver='latest')
    D = f.create_dataset("D", (nland,nnodes), data=D_np, dtype=datatype,compression="gzip")

    print("Total time to fill in D is %.5f sec" %(T.time()-start))
    
    # close
    f.close()


if graph_driver == 'igraph':
    # load from igraph
    G2 = Graph.Load(datadir + "igraph_graph",format="graphmlz")
    # G2 = Graph.Load("igraph_edgelist",format="edgelist")
    G2 = Graph.as_undirected(G2) # avoid inf when calculating distances
    nodes = G2.vs.indices
    nnodes = len(nodes)
    print("Nodes ids are in order = ", np.amax(nodes) + 1 == G2.vcount())

    # get node degrees for each index
    degrees = G2.vs.degree()
    node_degree_map = dict(zip(nodes,degrees))
    ndm_sorted_by_deg = {k: v for k,v in sorted(node_degree_map.items(), key=lambda x: x[1], reverse=True)}

    # randomly choose landmark ids
    # nland = 5
    landmark_ids = list(range(nnodes))
    # np.random.shuffle(landmark_ids)
    landmark_ids = np.sort(landmark_ids[:nland])
    landmark_map = dict(zip(list(range(nland)),landmark_ids))

    import collections

    print("Filling in data matrix...")
    start = T.time()
    f2 = h5py.File(datadir + "D.hdf5", "a",libver='latest')
    # clear data if already exists
    if 'D' in list(f2.keys()):
        del f2['D'] 
    d0 = Graph.shortest_paths(G2,landmark_ids[0])[0]
    d0 = np.array(d0,dtype=datatype)
    D = f2.create_dataset('D', (1,nnodes), data=d0, 
                maxshape=(nland,nnodes),dtype=datatype,
                compression="gzip",chunks=(nland,1000))
    # D_np = np.zeros((nland,nnodes),dtype=datatype)
    # D_np[0] = d0
    for i,n in enumerate(landmark_ids[1:]):
        # start2 = T.time()
        D.resize(D.shape[0] + 1, axis=0)  
        d = Graph.shortest_paths(G2,n)[0]
        # od = collections.OrderedDict(sorted(d.items()))
        D[-1] = np.array(d,dtype=datatype)
        if (i+2) % 5 == 0: 
            print("Iteration ", i+2,", shape = ", D.shape)
        # D_np[i+1] = list(d)
    # np.save("D_np.npy",D_np)

    # # use h5py to save and retrieve data
    # D = f2.create_dataset("D", (nland,nnodes), data=D_np, dtype=datatype,compression="gzip")

    print("Total time to fill in D is %.5f sec" %(T.time()-start))

    # close
    f2.close()

