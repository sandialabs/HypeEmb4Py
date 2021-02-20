### DEPRECATED

from numpy import *
from hypy_preproc import *
import os, pdb, time

nodes = 334863
landmarks = 100
min_dim = 4
max_dim = 4
nodes_per_run = 100000 # nodes per run of nonlandmark embedding
n_split = int(ceil((nodes-landmarks)/float(nodes_per_run)))
nprocs1 = 4 # parallel landmark procs
nprocs2 = 4 # parallel nonlandmark procs
nprocs3 = 4 # parallel validation (multiprocessing pool)
preprocess = 0 # do not split data

# set environment variables
# build mpi4py using openmpi-clang (install via port)?
mpicc = 'mpirun-openmpi-clang' 
dataloc = '/Users/kchowdh/Research/hypEmbedData' + '/' + "n" + str(nodes) + "_L" + str(landmarks)
savedir = '/Users/kchowdh/Research/hypEmbedData/results'

if preprocess == 1:
	# preprocess data for landmark and nonlandmark embedding
	os.system('mkdir ' + dataloc + '/temp')
	preprocdata(nodes, landmarks, nodes_per_run, dataloc, maxprocs=nprocs1, cleanup=True)
else:
	for dim in range(min_dim,max_dim+1):

		# perform landmark embedding in parallel using MPI
		start_land = time.time()
		landmark_input = landmark_preproc(nodes, landmarks, dim, dataloc)
		start = time.time()
		os.system(mpicc + ' -np ' + str(nprocs1) + ' python hypy_embed_landmarks_MPI.py' + landmark_input)
		# get best embedding 
		flandopt, land_coord, k, land_time, land_ind = gather_landmarks(nodes, landmarks, dim, dataloc, nprocs1)
		print "Complete landmark time is ", time.time() - start_land
		print "Best embedding is %f with time %f" %(flandopt, land_time)
		print land_ind

		# perform nonlandmark embedding using MPI as well
		start_nonland = time.time()
		for split in range(n_split):
			nonland_input = nonlandmark_preproc(nodes, landmarks, dim, split, dataloc)
			os.system(mpicc + ' -np ' + str(nprocs2) + ' python hypy_embed_nonlandmarks_MPI.py' + nonland_input)
			# gather solutoins per split
			gather_nonlandmarks_split(nodes, landmarks, dim, dataloc, split, nprocs2)
		C, nonland_avg_rel_error, hypy_nonland_time = gather_nonlandmarks(nodes, landmarks, dim, dataloc, n_split)
		tot_land_time = time.time() - start_nonland
		print "************ Total nonland time: ", tot_land_time, "*************"

		# run validation test (must create /results directory)
		print "begin validation tests..."
		val_input = validation_preproc(nodes, landmarks, dim, nprocs3, dataloc, savedir)
		print val_input
		os.system('python hypy_validation.py' + val_input)