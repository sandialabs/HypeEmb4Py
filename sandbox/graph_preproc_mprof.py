import numpy as np
import networkx as nx 
import time as T 
from igraph import Graph
import os, sys
from memory_profiler import profile

'''
This file imports an edge list from the SNAP database and then 
1. Checks connectedness and gets maximum subgraph of connected components
2. converts the graph labels into consecutive integers starting with 0
3. writes new graph to file
'''
# graph_driver = 'networkx'
graph_driver = 'igraph'
print("Using graph driver: ", graph_driver)

HOME = os.getenv("HOME")
datadir = HOME + "/Research/data/"
network = str(sys.argv[1])
edgelist = datadir + "com-" + network + ".ungraph.remapped.txt"


# if graph_driver == 'igraph':
print("\nStarting igraph preprocessing...")
start = T.time()
print("Loading %s ..." %edgelist)
G = Graph.Read(edgelist,format="edgelist", directed=False)
# print("Getting largest connected component...")
# C = Graph.components(G,mode="weak")
# del G
# G = C.subgraph(np.argmax(C.sizes()))
# del C
# G = G.vs.graph
# G = Graph.as_undirected(G)
print("Graph properties: ", G.vcount(),G.ecount())

# save graph
print("writing graph to file...")
G.write_graphmlz(datadir + "igraph_graph_" + network,compresslevel=3)
# G.write_edgelist("igraph_edgelist")

# # load graph
# G = Graph.Load("igraph_graph",format="graphmlz")
# # G = Graph.Load("igraph_edgelist",format="edgelist")
# nodes = G.vs.indices
# print(np.amax(nodes) + 1 == G.vcount())
print("Total time %.3f seconds" %(T.time()-start))

# # network x
# @profile
# def test():
#     if graph_driver == 'networkx':
#         # network x
#         # load graph
#         start = T.time()
#         print("Loading edge list to from graph...")
#         G = nx.read_edgelist(edgelist, nodetype=int)
#         print("Loading edge list took %.5f sec" %(T.time()-start))

#         # check if connected
#         print("Checking connectedness...")
#         # start = T.time()
#         if nx.is_connected(G) is False:
#             print("Graph is not connected. Getting largest connected component.")
#             largest_cc = max(nx.connected_components(G), key=len)
#             G = g.subgraph(largest_cc).copy()
#         else:
#             print("Graph is connected!")
#         print("Total time to check connectedness is %.5f sec" %(T.time()-start))

#         # get node ids
#         node_ids = np.sort(list(G.nodes()))
#         print("# of nodes = %i, # of edges = %i" 
#                 %(len(node_ids),G.number_of_edges()))

#         # create new mapping
#         # create_new_map = (len(node_ids)-1 == np.amax(node_ids))
#         # if create_new_map == True:
#         # start = T.time()
#         print("Converting nodes labels to consecutive integers...")
#         G = nx.convert_node_labels_to_integers(G)
#         print("Total time to remap is %.5f sec" %(T.time()-start))

#         # write new edge list
#         saved_edges = "networkx_edgelist"
#         print("Writing remapped and modified graph...")
#         # start = T.time()
#         f = open(datadir + saved_edges, "wb")
#         nx.write_edgelist(G,f,comments="Node->Node edges",data=False)
#         f.close()
#         print("Total time is %.5f sec" %(T.time()-start))

#         # # read saved graph
#         # print("Reading graph...")
#         # start = T.time()
#         # Gnew = nx.read_edgelist(saved_edges,nodetype=int)
#         # print("Total time is %.5f sec" %(T.time()-start))



# if __name__ == '__main__':
#     test()
