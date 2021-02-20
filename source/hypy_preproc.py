from numpy import *
from scipy.optimize import minimize, fmin_l_bfgs_b
import multiprocessing as mp
import sys, time, pdb, os
# import gc

# Parallel landmark and nonlandmark embedding
from hypyfun2 import *

def preprocdata(nodes, landmarks, nodes_per_run, dataloc, maxprocs4seed=1000, cleanup=False):
	print "\n", "*"*60
	print "Pre-processing: splitting D mtx for n = %i and L = %i" %(nodes, landmarks)
	print "*"*60

	if cleanup: clean(dataloc)
	
	# calculate the number of partitions for the D matrix
	n_split = int(ceil((nodes-landmarks)/float(nodes_per_run)))
	savetxt(dataloc + '/nonland_nsplit.txt', [n_split],fmt='%i')

	print "Splitting nonlandmark matrix and etc..."
	preproc(nodes, landmarks, dataloc, n_split, maxprocs4seed)

def landmark_preproc(nodes, landmarks, dim, dataloc):
	input_var = [" -n "," -L "," -d "," -s "]
	input_val = [nodes, landmarks, dim, dataloc]
	input_tot = ""
	for i,var in enumerate(input_var):
		input_tot += var + str(input_val[i])
	return input_tot

def nonlandmark_preproc(nodes, landmarks, dim, split, dataloc):
	input_var = [" -n "," -L "," -d "," -sp ", " -s "]
	input_val = [nodes, landmarks, dim, split, dataloc]
	input_tot = ""
	os.system('mkdir ' + dataloc + '/temp')
	for i,var in enumerate(input_var):
		input_tot += var + str(input_val[i])
	return input_tot

def validation_preproc(nodes, landmarks, dim, nproc, dataloc, savedir):
	input_var = [" -n "," -L "," -d "," -p ", " -s1 ", " -s2 "]
	input_val = [nodes, landmarks, dim, nproc, dataloc, savedir]
	input_tot = ""
	for i,var in enumerate(input_var):
		input_tot += var + str(input_val[i])
	return input_tot