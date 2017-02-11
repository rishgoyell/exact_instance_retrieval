#takes as input data to be clustered(n*d) and returns cluster IDs(n*knn) and dist from those cluster centers(n*knn)
#manipulate knn to give desired number of nn

import numpy as np
from pyflann import *

def clusterAssign(X,knn,centers):
	#print "loading centers............"
	#centers = np.load('matrices/data30/codebook15000_30.npy')

	flann = FLANN()
	print "Building Index..........."
	params = flann.build_index(centers, algorithm='autotuned', target_precision=0.98, build_weight = 0.00, memory_weight=0.00, sample_fraction = 0.9)
	print "Built!!!........"

	X = X.astype('float32')

	print "assigning clusters............"
	clusterID, dist = flann.nn_index(X, knn, checks=params["checks"])
	print "Done!"
	return clusterID, dist
