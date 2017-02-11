import numpy as np
from pyflann import *
import math
import clusterAssign
import fileStructure
import tfidf
import ml_metrics as metrics
import hamEmbed

he_threshold = 52 

invFileTable = np.load('matrices/data72/invFileTable20000k3.npy')
centers = np.load('matrices/data72/codebook.npy')
numcenters = invFileTable.shape[0]

#removing a percentage of the most popular words from list
del_rows_sum = np.sum(invFileTable, axis=1)
del_rows = np.argsort(-del_rows_sum)
num_del_rows1 = int(0.85*numcenters)		#hyperparameter to prune off very popular features
num_del_rows2 = int(1*numcenters)		#hyperparameters to prune off unpopular features
del_rows1 = del_rows[:num_del_rows1]
del_rows2 = del_rows[num_del_rows2:]
del_rows = np.concatenate((del_rows1,del_rows2))

del_rows = np.sort(del_rows)
row_index = -np.ones(numcenters)
del_row_index = 0
good_row_count = 0

for i in range(numcenters):
    if del_row_index < del_rows.shape[0] and del_rows[del_row_index] == i:
        del_row_index = del_row_index+1
    else:
        row_index[i] = good_row_count
        good_row_count = good_row_count+1


invFileTable = np.delete(invFileTable,del_rows, axis=0)
row_index = row_index.astype('int')


#itf-df for all words
idftrain, invFileTable= tfidf.tfidf(invFileTable)

print "Loading test data..........."
Xtest = np.loadtxt('tempX.txt', delimiter=',')
Ytest = np.loadtxt('tempY.txt', delimiter=',')

print "Assigning Clusters........"
testClusterID, testDist = clusterAssign.clusterAssign(Xtest,1,centers)

print "Computing corresponding vectors......."
testStructure = fileStructure.imgFeatures(Ytest, testClusterID, testDist, centers)

#removing rows corresponding to poular words
testStructure = np.delete(testStructure,del_rows, axis=0)

print "Computing TF-IDF score.........."
idftest, testStructure = tfidf.tfidf(testStructure)


print "loading medians and embeddings......."
median = np.load('matrices/data72/median72.npy')
npz = np.load('matrices/data72/hamEmb72.npz')
ML = npz['arr_0']
CIL = npz['arr_1']

print "computing hamming embeddings of test data...."
HE = hamEmbed.binEmbeddings(Xtest, testClusterID, median, 1)

numTestImgs = Ytest[-1]
numTrainImgs = invFileTable.shape[1] 
numTestFeatures = Xtest.shape[0]

scores = np.zeros([numTestImgs,numTrainImgs])		#matrix to store scores

print "computing scores for each test image......."
for i in range(numTestFeatures):
    tfc = row_index[testClusterID[i]]
    if tfc==-1:
        continue
    indm = ML[testClusterID[i]]
    #iterating over features assigned to corresponding clusterAssign
    for j in range(np.shape(indm)[0]):
        g = np.logical_xor(HE[i,:],indm[j,:])
        if np.sum(g) < he_threshold:
            indv = CIL[testClusterID[i]][j]-1
            scores[Ytest[i]-1, indv] = scores[Ytest[i]-1, indv] + idftrain[tfc] * idftest[tfc]#testStructure[tfc,Ytest[i]-1] *trainStructure[tfc,indv]

scores = np.divide(scores,np.sum(invFileTable, axis=0))
ranks = np.argsort(-scores, axis=1)

for i in range(numTestImgs):
    print i/testImgsperFolder+1,">>>>>", ranks[i,:10]/trainImgsperFolder+1

with open('folders.txt') as f:
	folders = f.readlines()
folders = [x.strip() for x in folders]

with open('images.txt') as f:
	images = f.readlines()
images = [x.strip() for x in images]

filename = 'test$.txt'
for i in 1+range(numTestImgs):
	fileID = open(filename.replace("$",str(i)),'w')
	for j in range(numTrainImgs):
		fileID.write(images[j],folders[j/72])
	fileID.close()


