import numpy as np 
import matplotlib.pyplot as mpl
import os, pdb, time, pickle
import hypy_preproc as hypepp
# requires multiprocess, pandas and mpi4py

class GraphAlg:
	def __init__(self,name,datadir=None):
		self.name = name
		self.datadir = datadir
		self.mpicc = None
		if datadir!=None and os.path.exists(datadir + "metadata.pkl"):
			self.metadata = self.load_meta("metadata")
	def set_main_env(self,maxprocs4seed,split,nodes_per_run):
		'''
		Split the nonlandmark embedding into serial chunks

		If split = True, then user has to define nodes_per_run. Split
		defines how many serial chunks to perform the nonlandmark
		embedding. Each chunk is then performed in parallel. E.g., 
		if nnodes = 100 and nodes per run is 2, then the first 50
		nodes are embedding in parallel, followed by the next 50.

		Most data is written to the datadir so that there is minimal
		sharing of data and each function can be run at different times.  
		'''
		# set hyperbolic dimension and processors for nonland embedding
		nodes = self.metadata['nnodes']
		landmarks = self.metadata['nL']
		datadir = self.metadata['datadir']
		self._nprocs1 = maxprocs4seed # parallel landmark procs
		# self._nprocs2 = nprocs # parallel nonlandmark procs
		# self._nprocs3 = nprocs # parallel validation (multiprocessing pool)

		if split==True:
			nodes_per_run = 100000 # nodes per run of nonlandmark embedding based on 
			self.nl_split = int(np.ceil((nodes-landmarks)/float(nodes_per_run)))
			# preprocess data for landmark and nonlandmark embedding
			os.system('mkdir ' + datadir + '/temp')
			hypepp.preprocdata(nodes, landmarks, nodes_per_run, datadir, maxprocs4seed=100*self._nprocs1, cleanup=True)

	def embed_landmarks(self,hdim,mpicc,nrepeat):
		MD = self.metadata
		self.metadata['hdim'] = hdim
		print "\nHyperbolic dimension for landmark embedding is %i." %(hdim)
		# perform landmark embedding in parallel using MPI
		start_land = time.time()
		landmark_input = hypepp.landmark_preproc(MD['nnodes'], MD['nL'], hdim, MD['datadir'])
		start = time.time()
		self.mpicc = mpicc
		os.system(mpicc + ' --oversubscribe -np ' + str(nrepeat) + ' python hypy_embed_landmarks_MPI.py' + landmark_input)
		# get best embedding 
		flandopt, land_coord, k, land_time, land_ind = hypepp.gather_landmarks(MD['nnodes'], MD['nL'], hdim, MD['datadir'], nrepeat)
		self.land_coord = land_coord
		self.full_land_coord = self.get_full_coordinates(self.land_coord)
		print "Complete landmark time is ", time.time() - start_land
		print "Best embedding error is %f" %(flandopt)
		# print land_ind
		return land_coord, 1.0*k, self.full_land_coord

	def embed_nonlandmarks(self,nprocs2):
		MD = self.metadata
		nodes = MD['nnodes']
		landmarks = MD['nL']
		datadir = MD['datadir']
		dim = MD['hdim']
		print "\nHyperbolic dimension for non-landmark embedding is %i." %(dim)
		# perform nonlandmark embedding using MPI as well
		start_nonland = time.time()
		print "Number of batches of nonlandmarks is %i" %(self.nl_split)
		for split in range(self.nl_split): # nonlandmark splits
			nonland_input = hypepp.nonlandmark_preproc(nodes, landmarks, dim, split, datadir)
			os.system(self.mpicc + ' -np ' + str(nprocs2) + ' python hypy_embed_nonlandmarks_MPI.py' + nonland_input)
			# gather solutoins per split
			hypepp.gather_nonlandmarks_split(nodes, landmarks, dim, datadir, split, nprocs2)
		C, nonland_avg_rel_error, hypy_nonland_time = hypepp.gather_nonlandmarks(nodes, landmarks, dim, datadir, self.nl_split)
		tot_land_time = time.time() - start_nonland
		print "************ Total nonland time: ", tot_land_time, "*************"

	def get_full_coordinates(self,coordinates):
		''' Given a 2d array of d-dim hyper coordinates
		return the full d+1-dim coordinate for plotting
		'''
		u_dp1 = np.sqrt(np.sum(coordinates**2,1)+1)
		full_coord = np.hstack((coordinates,np.atleast_2d(u_dp1).T))
		return full_coord

	def plot_landmarks(self,P):
		dim = P.shape[1]
		if dim > 3: print "Only for use when dim 2 or 3"
		# perform test to make sure we are on the hyperboloid
		p1 = P[0]
		if sum(p1[:-1]**2) - p1[-1]**2 != -1.0: print "Need to compute additional dimension."
		else:
			if dim == 2:
				fig = mpl.figure()
				ax = fig.add_subplot(1,1,1)
				ax.plot(P[:,0],P[:,1],'.')
			return fig
			if dim == 3:
				import matplotlib.pyplot as plt
				from mpl_toolkits.mplot3d import Axes3D
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')

	def ReMapNodes(self,filename,location,nprocs):
		'''
		Remap the nodes given an edge list file
		into consecutive numbers. This is necessary
		for SNAP data sets where indices could be greater
		than the actual number of nodes

		Requires pandas and multiprocessing package 
		'''
		os.system('python map_nodes_new.py' + ' -nprocs=' + str(nprocs) \
											+ ' -filename=' + filename  \
											+ ' -location=' + location)
		self.edge_list_file = filename + '_remapped'
		self.edge_list_location = location

	def GenAdjMat(self,filename,location,savedir,nL,nprocs=4,nval=100):
		'''
		Compute the landmarks given the edge list along
		with distances from nodes to each landmark
		'''
		self.nL = nL # set the number of landmarks
		self.ntest = nval # set the number of test pairs
		self.datadir = savedir # data location for hyp emd algorithm
		self.metadata = {"nL":self.nL, "ntest":self.ntest, "datadir":self.datadir}

		os.system('python gen_adj_mtx_new.py' + ' -filename=' + filename  \
											+ ' -location=' + location  \
											+ ' -savedir=' + savedir   \
											+ ' -nprocs=' + str(nprocs) \
											+ ' -nL=' + str(nL) \
											+ ' -nval=' + str(self.ntest) )
		self.nnodes = np.int(np.loadtxt(self.datadir + 'nNodes.txt'))
		self.metadata["nnodes"] = self.nnodes
		self.save_meta(self.metadata,"metadata")

	def validate(self,savedir,nprocs3):
		# validate and save validation meta data in savedir
		MD = self.metadata
		nodes = MD['nnodes']
		landmarks = MD['nL']
		datadir = MD['datadir']
		dim = MD['hdim']
		# run validation test (must create /results directory)
		print "begin validation tests..."
		val_input = hypepp.validation_preproc(nodes, landmarks, dim, nprocs3, datadir, savedir)
		# print val_input
		os.system('python hypy_validation.py' + val_input)

	def clean_datadir(self):
		os.system('rm ' + self.datadir + '*rank*')
		os.system('rm ' + self.datadir + '*error*')
		os.system('rm ' + self.datadir + '*opt*')
		os.system('rm ' + self.datadir + 'hypy_curv.txt')
		os.system('rm -rf ' + self.datadir + 'temp')

	def save_meta(self,obj,name):
	    with open(self.datadir + name + '.pkl', 'wb') as f:
	    	pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

	def load_meta(self,name):
	    with open(self.datadir + name + '.pkl', 'rb') as f:
	    	return pickle.load(f)



def test_remap_nodes(snapdir="../data/amazon/snap/"):
	''' Instantiate with preprocessing of graph '''
	# Instantiate graph object
	G = GraphAlg("amazon")

	# Renumber gradph edges to be consecutive
	G.ReMapNodes(filename="amazon_edges",location=snapdir,nprocs=4)

def test_create_adj_matrix(snapdir="../data/amazon/snap/",savedir="../data/amazon/"):
	''' Instantiate with preprocessing of landmarks '''
	# Instantiate graph object
	G = GraphAlg("amazon")

	# obtain landmark nodes and adjacency matrix
	G.GenAdjMat(filename="amazon_edges_remapped",location=snapdir,savedir=savedir,nL=100,nprocs=4,nval=1000)

def test_embedding(datadir="../data/amazon/"):
	# '''Run without adj matrix generation '''

	# Instantiate graph object with previously run adj matrix
	G = GraphAlg("amazon",datadir=datadir)

	# set main environment and preprocess for splitting nonlandmarks
	# this step can be done once for multiple hyperbolic dimension tests
	# this does not depend on the landmarks, only the number of nonlandmarks
	G.set_main_env(maxprocs4seed=8,split=True,nodes_per_run=100000)

	# perform landmark embedding
	L, k, L0 = G.embed_landmarks(hdim=4,mpicc='mpirun-openmpi-clang',nrepeat=8)

	G.embed_nonlandmarks(nprocs2=4)

	G.validate(savedir=datadir,nprocs3=4)







