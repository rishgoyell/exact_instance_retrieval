#takes as input the data to be clustered and returns an array of cluster centers using approximate kmenas algorithm
#requires FLANN library's python bindings

import numpy as np
from pyflann import *
import timeit

def akm(X, numclusters, numiters):
	assert X.shape[1]==128
	numpts = X.shape[0]		#number of points
	#numclusters = 15000		#number of clusters 
	#numiters = 50			#number of iterations
	X = X.astype('float32')

	#randomly initialising centers from among datapoints
	randomperm = np.random.permutation(numpts)
	randomperm = randomperm[0:numclusters]
	centers = X[randomperm,:]

	#initialising FLANN instance
	flann = FLANN()

	iter_err = np.zeros(numiters)			#error for each iteration
	ptid = np.zeros(numpts)					#cluster id for each point in an iteration
	stopcriteria = 0.000001
	olderr = 10000						#initialisation
	loadFlag = 1

	#iterative updates begin
	for i in range(numiters):
		start_time = timeit.default_timer()
		tmpcenters = np.zeros(centers.shape)	#sum of points in a cluster
		clusterSize = np.zeros(numclusters)		#number of points in a cluster
		err = 0

		#building index
		print "Building Index for iteration", (i+1)  
		params = flann.build_index(centers, algorithm="autotuned", target_precision=0.70, build_weight = 0.0, memory_weight=0.0, sample_fraction = 0.8)
		old_centers = centers 		#Create a copy of centers
		del centers

		#update cluster centers and assign each point to a cluster
		for j in range(numpts):
			if j%100000==0:
				print j,"out of",numpts
			singlept = X[j,:]
			result, dist = flann.nn_index(singlept,1, checks=params["checks"]);
			tmpcenters[result,:] = singlept.astype('float32') + tmpcenters[result,:]
			clusterSize[result] = clusterSize[result]+1
			ptid[j] = result
			err = err + dist
		centers = np.zeros([numclusters,128]);
		centers = centers.astype('float32')
		for k in range(numclusters):
			if clusterSize[k] > 0:
				centers[k, :] = tmpcenters[k, :]/clusterSize[k]

		#Error value is total squared distance from cluster centers
		iter_err[i] = err
		print "Cycle:",(i+1),"error:",err

		 #test for termination
		if i>1:
			if abs(olderr-err)/iter_err[1]<stopcriteria:
				print olderr
				return old_centers
		olderr = err
		elapsed_time = timeit.default_timer() - start_time
		print elapsed_time

	return old_centers
			    
	



