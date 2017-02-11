import numpy as np
from pyflann import *
import math
import clusterAssign
import fileStructure
import tfidf
import ml_metrics as metrics

invFileTable = np.load('matrices/data50/invFileTable20000k3.npy')
centers = np.load('matrices/data50/codebook.npy')

#removing a percentage of the most popular words from list
del_rows_sum = np.sum(invFileTable, axis=1)
del_rows = np.argsort(-del_rows_sum)
num_del_rows1 = int(0.85*invFileTable.shape[0])		#hyperparameter to prune off very popular features
num_del_rows2 = int(1*invFileTable.shape[0])		#hyperparameters to prune off unpopular features
del_rows1 = del_rows[:num_del_rows1]
del_rows2 = del_rows[num_del_rows2:]
del_rows = np.concatenate((del_rows1,del_rows2))

invFileTable = np.delete(invFileTable,del_rows, axis=0)

#itf-df for all words
invFileTable = tfidf.tfidf(invFileTable)
normaliseS = np.linalg.norm(invFileTable, axis=0)
invFileTable = np.divide(invFileTable,normaliseS)
np.nan_to_num(invFileTable)

#print "Loading test data..........."
#Xtest = np.loadtxt('matrices/testX5.txt', delimiter=',')
#Ytest = np.loadtxt('matrices/testY5.txt', dtype=int, delimiter=',')
#Xtest = np.transpose(Xtest)

#print "Assigning Clusters........"
#testClusterID, testDist = clusterAssign.clusterAssign(Xtest,1,centers)

print "Computing corresponding vectors......."
testStructure = fileStructure.imgFeatures(Ytest, testClusterID, testDist,centers)

#removing rows corresponding to poular words
testStructure = np.delete(testStructure,del_rows, axis=0)

print "Computing TF-IDF score.........."
testStructure = tfidf.tfidf(testStructure)

normaliseS = np.linalg.norm(testStructure, axis=0)
testStructure = np.divide(testStructure,normaliseS)
np.nan_to_num(testStructure)


print "Ranking......."
testStructure = np.transpose(testStructure)
scores = np.matmul(testStructure, invFileTable)
ranks = np.argsort(-scores, axis=1)

numImgs = ranks.shape[0]
mapest = 0; 
trainImgsperFolder = invFileTable.shape[1]/84
testImgsperFolder = numImgs/84

for i in range(numImgs):
	x = (i/testImgsperFolder)*trainImgsperFolder
	mapest = mapest + metrics.apk(range(x, x+trainImgsperFolder), ranks[i,:], 10)

mapest = mapest/numImgs
print mapest
#for i in range(numImgs):
#	print i/testImgsperFolder+1,">>>>>", ranks[i,:10]/trainImgsperFolder+1















