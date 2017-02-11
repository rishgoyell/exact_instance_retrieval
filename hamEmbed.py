from pyflann import *
import numpy as np
import os.path

def hamEmbed():
        
        d=128
        d_b=128
	#<<<<<<<<<<<<<<<<<<<<<<<<<<<<computing median matrix to act as thresholds>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	if os.path.isfile('matrices/data72/projection.npy'):
		P = np.load('matrices/data72/projection.npy')
	else:
		#compute projection matrix
		M = np.random.randn(d,d)
		Q,R = np.linalg.qr(M)
		P = Q[:d_b, :]
		np.save('matrices/data72/projection.npy',P)

	#load data matrix and take projections
	X = np.load('matrices/data72/Xf.npy')
        assignments = np.load('matrices/data72/clusterID20000k3.npy')
	proj = np.matmul(P, np.transpose(X))		#128*number of features matrix
        print "Loading centers........."
	#load centers
	centers = np.load('matrices/data72/codebook.npy')
	centers = np.transpose(centers)

	numpts = X.shape[0]
	#if assignments.ndim != 1:
	#	assignments = assignments[:,1]

	k = centers.shape[1]	#codebook size
	median_matrix = np.zeros([d_b,k])		#num_projected_features*num_visual_words
        print "calculating medians......"
	for i in range(k):
		pos = np.where(assignments==i)[0]
		if pos.shape[0]==0:
			median_matrix[:, i] = centers[:, i]
		else:
			temp_matrix = proj[:, pos]
			median_matrix[:,i] = np.median(temp_matrix, axis = 1)
	print "Done computing medians, now saving...."
        np.save('matrices/data72/median72.npy', median_matrix)
        print "calling binEmbeddings...."
	ml, cil = binEmbeddings(X, assignments, median_matrix, 0)
	np.savez('matrices/data72/hamEmb72.npz',ml,cil)


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<computing binary representations>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#input: data for which binary embeddings are to be learnt, centers from codebook, cluster assignments of concerned points, median matrix

def binEmbeddings(X, assignments, median_matrix, flag):
	
        print median_matrix.shape
	P = np.load('matrices/data72/projection.npy')
	numclusters = median_matrix.shape[1]
	if assignments.ndim==1:
		knn=1
	else:
		knn = assignments.shape[1]

	matList = [None]*numclusters
	corrImgList = [None]*numclusters
	X = np.matmul(X, np.transpose(P))
	del P
	if flag == 0:
                Y = np.load('matrices/data72/Yf.npy')
		print "computing embeddings"
		for i in range(numclusters):
                        if i%1000==0:
			    print "Working on Cluster ", i
			a = np.where(assignments==i)[0]
			matList[i] = np.zeros([a.shape[0],128], dtype=bool)
			corrImgList[i] = np.zeros([a.shape[0]],dtype=int)
			count = 0
			temp_med = median_matrix[:,i]
			for j in a:
				matList[i][count,:] = ((X[j,:]- np.transpose(temp_med))>0)
				corrImgList[i][count] =Y[j] 
				count = count+1

		return matList, corrImgList
	else:
		for i in range(X.shape[0]):
                        if i%10000==0:
                            print "Working on test data ", i
			X[i,:] = ((X[i,:]-np.transpose(median_matrix[:,assignments[i]]))>0)

		return X





	





