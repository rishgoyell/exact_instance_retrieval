#takes as input cluster centers, cluster ids nn and distances from them for all points
#outputs inverted file structure
import numpy as np

def imgWord(object):
	def __init__(self,imgid, vwcount):
		self.imgid = imgid
		self.vwcount = vwcount

def imgFeatures(Y, clusterIDs, dist, centers):
	# print "Loading Y"
	# Y = np.loadtxt('matrices/Y30.txt', dtype=int, delimiter=',')
	#print "Loading centers"
	#centers = np.load('matrices/data30/codebook15000_30.npy')
	# print "Loading cluster IDs"
	# clusterIDs = np.load('matrices/assignment30.npy')
	# print "Loading distances from center"
	# dist = np.load('matrices/dist30.npy')

	numclusters = centers.shape[0]
	if clusterIDs.ndim==1:
		knn=1
	else:
		knn = clusterIDs.shape[1]
	
	numpts = clusterIDs.shape[0]
	numimgs = Y[-1]
	count = 0
	print numclusters,numimgs
	invFileTable = np.zeros([numclusters, numimgs])
	
	#calculating weights from distances by exponentiaition
	if knn!=1:
		dist[np.where(dist==0)] = 0.000001
		expdist = np.square(np.reciprocal(dist))
		tempsum = np.sum(expdist,axis=1)
		#assert (tempsum.shape[1] == 1) print "check behaviour of np.sum"
		for i in range(knn):
			expdist[:,i] = np.divide(expdist[:,i],tempsum)
		#assert (expdist.shape == [numpts,knn]) print "check dimension of expdist"
		for i in range(Y.shape[0]):
			for j in range(knn):
				count=count+1
				invFileTable[clusterIDs[i,j],Y[i]-1] = invFileTable[clusterIDs[i,j],Y[i]-1] + expdist[i,j]
	else:
		expdist = np.ones(dist.shape[0])
		for i in range(Y.shape[0]):
			count=count+1
			invFileTable[clusterIDs[i],Y[i]-1] = invFileTable[clusterIDs[i],Y[i]-1] + expdist[i]

	#np.save('invFileTable15000x30', invFileTable)
	print count
	return invFileTable
	
