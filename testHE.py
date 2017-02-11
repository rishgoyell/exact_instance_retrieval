# import numpy as np
# from pyflann import *
# import math
# import clusterAssign
# import fileStructure
# import tfidf
# import ml_metrics as metrics
# import hamEmbed

# he_threshold = 52 

# invFileTable = np.load('matrices/data72/invFileTable20000k3.npy')
# centers = np.load('matrices/data72/codebook.npy')
# numcenters = invFileTable.shape[0]

# print "<<<<<<<<<<", invFileTable.shape," >>>>>>>>>>>>>>>"

# #removing a percentage of the most popular words from list
# del_rows_sum = np.sum(invFileTable, axis=1)
# del_rows = np.argsort(-del_rows_sum)
# num_del_rows1 = int(0.85*numcenters)		#hyperparameter to prune off very popular features
# num_del_rows2 = int(1*numcenters)		#hyperparameters to prune off unpopular features
# del_rows1 = del_rows[:num_del_rows1]
# del_rows2 = del_rows[num_del_rows2:]
# del_rows = np.concatenate((del_rows1,del_rows2))

# del_rows = np.sort(del_rows)
# row_index = -np.ones(numcenters)
# del_row_index = 0
# good_row_count = 0

# for i in range(numcenters):
#     if del_row_index < del_rows.shape[0] and del_rows[del_row_index] == i:
#         del_row_index = del_row_index+1
#     else:
#         row_index[i] = good_row_count
#         good_row_count = good_row_count+1


# invFileTable = np.delete(invFileTable,del_rows, axis=0)
# row_index = row_index.astype('int')

# #itf-df for all words
# idftrain, invFileTable= tfidf.tfidf(invFileTable)
# normaliseS = np.linalg.norm(invFileTable, axis=0)
# invFileTable = np.divide(invFileTable,normaliseS)
# np.nan_to_num(invFileTable)


# # print "Loading test data..........."
# # Xtest = np.load('matrices/testX5.npy')
# # Ytest = np.load('matrices/testY5.npy')
# #Xtest = np.transpose(Xtest)

# # print "Assigning Clusters........"
# # testClusterID, testDist = clusterAssign.clusterAssign(Xtest,1,centers)

# print "Computing corresponding vectors......."
# testStructure = fileStructure.imgFeatures(Ytest, testClusterID, testDist, centers)

# #removing rows corresponding to poular words
# testStructure = np.delete(testStructure,del_rows, axis=0)

# print "Computing TF-IDF score.........."
# idftest, testStructure = tfidf.tfidf(testStructure)

# normaliseS = np.linalg.norm(testStructure, axis=0)
# testStructure = np.divide(testStructure,normaliseS)
# np.nan_to_num(testStructure)


# print "loading medians and embeddings......."
# median = np.load('matrices/data72/median72.npy')
# npz = np.load('matrices/data72/hamEmb72.npz')
# ML = npz['arr_0']
# CIL = npz['arr_1']

# print "computing hamming embeddings of test data...."
# HE = hamEmbed.binEmbeddings(Xtest, testClusterID, median, 1)

# del median

# numTestImgs = Ytest[-1] 
# numTrainImgs = invFileTable.shape[1] 
# numTestFeatures = Xtest.shape[0]

# #norm1 = np.zeros([numTestImgs,numTrainImgs])		#matrix to store scores
# #norm2 = np.zeros([numTestImgs,numTrainImgs])		#matrix to store scores
# scores = np.zeros([numTestImgs,numTrainImgs])		#matrix to store scores
# nmcount = 0
# incount = 0.0
# tcount = 0.0
print "computing scores for each test image......."
for i in range(numTestFeatures):
    tfc = row_index[testClusterID[i]]
    if tfc==-1:
        continue
    indm = ML[testClusterID[i]]
    #iterating over features assigned to corresponding clusterAssign
    for j in range(np.shape(indm)[0]):
        tcount = tcount + 1
        g = np.logical_xor(HE[i,:],indm[j,:])
        if np.sum(g) < he_threshold:
            incount = incount + 1
            indv = CIL[testClusterID[i]][j]-1
            if Ytest[i]/5!=indv/72:
                nmcount= nmcount+1
            scores[Ytest[i]-1, indv] = scores[Ytest[i]-1, indv] + idftrain[tfc] * idftest[tfc]#testStructure[tfc,Ytest[i]-1] *trainStructure[tfc,indv]
            #norm1[Ytest[i]-1, indv] = norm1[Ytest[i]-1, indv] + np.square(testStructure[tfc,Ytest[i]-1])
            #norm2[Ytest[i]-1, indv] = norm2[Ytest[i]-1, indv] + np.square(invFileTable[tfc,indv])

#norm2[norm2==0]=1
#norm1[norm1==0]=1
#scores = np.divide(scores, np.sqrt(norm1))
scores = np.divide(scores,np.sum(invFileTable, axis=0))
ranks = np.argsort(-scores, axis=1)

numImgs = ranks.shape[0]
mapest = np.zeros([5,1]);
k = [5,10,30,72,invFileTable.shape[1]]
trainImgsperFolder = invFileTable.shape[1]/84
testImgsperFolder = numImgs/84

for i in range(numImgs):
    x = (i/testImgsperFolder)*trainImgsperFolder
    for j in range(5)
        mapest[j] = mapest[j] + metrics.apk(range(x, x+trainImgsperFolder), ranks[i,:], k[j])

mapest = mapest/numImgs

print mapest
for i in range(numTestImgs):
    print i/testImgsperFolder+1,">>>>>", ranks[i,:10]/trainImgsperFolder+1

print nmcount/tcount, "<<<<<<<>>>>>>>>", nmcount/incount

		

