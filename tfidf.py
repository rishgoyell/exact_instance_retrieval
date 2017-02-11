#input matrix v*i with v  number of visual words and i images
#output matrix of same size, with corresponding tfidf entries

import numpy as np

def tfidf(vwbyimg):
	numClusters = vwbyimg.shape[0]
	N = vwbyimg.shape[1]
	total_words = np.sum(vwbyimg,axis=0)
        idf = np.zeros(numClusters)

	assert total_words.any() != 0

	for i in range(numClusters):
		x = np.count_nonzero(vwbyimg[i,:])
		if x==0:
			IDF = 0
		else:
			IDF = np.log(N/x)
                        idf[i] = IDF
		assert vwbyimg[i,:].shape == total_words.shape
		vwbyimg[i,:] = np.divide(vwbyimg[i,:],total_words) * IDF
	return idf, vwbyimg
