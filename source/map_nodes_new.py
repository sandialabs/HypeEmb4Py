import numpy as np
import multiprocessing as mp
import time, pdb, sys
import pandas as pd
import argparse

'''Renumbering graph in sequential order using multi threading'''

'''
Updates 10-2020:

Exploring use of networkx, which has a more intuitive interface
And perhaps uncessary to renumber lists
Example:
# load graph from edge list
g = nx.read_edgelist("com-youtube.ungraph.txt",nodetype=int)
# get number of nodes
nnodes = len(g.nodes())
# get degrees
degrees = [nd[1] for nd in g.degree]
# check and remove if there are any zero degree edges 
assert amin(degrees) > 0
# check if connected
assert nx.is_connected(g) == True
# if not, get largest component and get resulting subgraph (slow)
largest_cc = max(nx.connected_components(g), key=len)
g_cc = g.subgraph(largest_cc).copy()
# to relabel nodes, create a mapping dictionary from existin
#	nodes to new labels and then nx.relabel_nodes(g,mapping)
'''

def split(seq, procs):
	# seq is a list of input parameters
	# procs are the number of processors
	avg = len(seq) / float(procs)
	out = []
	last = 0.0
	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg
	return out

def load_graph_data(filename):
	print 'Loading data...'
	e0 = pd.read_csv(filename,sep='\t',index_col=0,header=None,skiprows=4,dtype='int32')
	e1 = np.array(e0.index,dtype='int32')
	e1.shape = (len(e1),1)
	e2 = np.array(e0,dtype='int32')
	edge_list = np.hstack((e1,e2)).astype('int32')
	n_edges = len(edge_list)

	# get number of nodes
	nodes = np.unique(edge_list.flatten())
	print 'Done loading data!'

	return edge_list, nodes, n_edges

def get_new_edge_list(edge_list_temp):
	''' renumber edge list using dictionary of indices, D '''
	new_edge_list = np.zeros(edge_list_temp.shape,dtype='int32')
	for j,r in enumerate(edge_list_temp):
		if j % 100000 == 0: print "%.2f percent done" %(float(j)/len(edge_list_temp))
		new_edge_list[j,0] = D[r[0]]
		new_edge_list[j,1] = D[r[1]]
	return new_edge_list

def remap_edge_list(edge_list, nprocs=4):
	''' re-map edge list to have consecutive numbers (in parallel)'''
	# create a dictionary which maps node # to index 
	# note that the node # might be higher than the total number of 
	# unique n_nodes, which is why we are doing this

	sp = split(range(len(edge_list)),nprocs)
	edge_split = [edge_list[s,:] for s in sp]
	pool = mp.Pool(processes=nprocs)
	new_edge_lists = pool.map(get_new_edge_list,edge_split)
	new_edge_list = np.vstack(new_edge_lists)

	return new_edge_list



#######################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-nprocs", type=int, help="number of processors")
parser.add_argument("-filename", type=str, help="name of original snap file (string)")
parser.add_argument("-location", type=str, help="source location of snap data file (string)")

args = parser.parse_args()
#######################################################################


# load data via pandas read_csv function first
graphsrc = args.filename #original file from SNAP
nprocs = args.nprocs #processes

dirloc = args.location
datafile = dirloc + graphsrc
newfilename = dirloc + graphsrc + '_remapped' # remapped edge list

edge_list, nodes, n_edges = load_graph_data(datafile)

# create dictionary map from old nodes to new nodes
D = {}
for i,n in enumerate(nodes):
	D[n] = i

# remap function requires dictionary D as a global
start = time.time()
new_edge_list = remap_edge_list(edge_list,nprocs)
end = time.time()
print "Total time for remapping is %f seconds" %(end-start)

# save data
print 'writing to text file...'
np.savetxt(newfilename,new_edge_list,fmt='%i',delimiter='\t')

