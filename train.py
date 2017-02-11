import numpy as np
import fileStructure
import clusterAssign
import akm
import hamEmbed

#print "Loading Data..............."
#X1 = np.loadtxt('matrices/data72/Xf1.txt',delimiter=',')
#X2 = np.loadtxt('matrices/data72/Xf2.txt',delimiter=',')
#print "Data Loaded!"

#X = np.concatenate((X1, X2))
#assert X.shape[0]==X1.shape[0]+X2.shape[0]

#del X1
#del X2

print "Saving X as np array"
np.save('matrices/data72/Xf.npy',X)

numclusters = 20000
numiters = 50

print "running approx k-means........."
print "Take a break, this may take a lot of time..........."
centers = akm.akm(X, numclusters, numiters)

print "Saving codebook......."
np.save('matrices/data72/codebook.npy', centers)

print "Assigning Clusters........."
clusterID, dist = clusterAssign.clusterAssign(X, 3, centers)

print "Loading Y matrix........."
Y1 = np.loadtxt('matrices/data72/Yf1.txt',dtype = int, delimiter=',')
Y2 = np.loadtxt('matrices/data72/Yf2.txt',dtype = int, delimiter=',')

Y = np.concatenate((Y1,Y2))
assert Y.shape[0]==Y1.shape[0]+Y2.shape[0]
del Y1,Y2

print "Saving Y as np array"
np.save('matrices/data72/Yf.npy',Y)

print "building file structure"
invFileTable = fileStructure.imgFeatures(Y, clusterID, dist, centers)

np.save('matrices/data72/invFileTable20000k3.npy', invFileTable)
np.save('matrices/data72/clusterID20000k3.npy', clusterID)

print "Computing Hamming Embeddings"
hamEmbed.hamEmbed()

print "Finished!!!"
