This was done as an assignment for a course on visual recognition.
The implementation is done in MATLAB and Python.
Code for SIFT feature extraction(taken from David Lowe's homepage)
Python dependencies required are Numpy, ml_metrics and FLANN
Read report.pdf for details on the algorithm

* The folder feature extraction contains code for SIFT feature extraction and conversion to rootSIFT.
* train.py: driver code for calling functions in logical order to learn a model given data
* akm.py: contains code for approximate k-means function
* clusterAssign.py: function builds index of learnt centers and assigns a visual word to any feature
* fileStructure.py: computes tf-idf score of a set of features
* hamEmbed.py: contains two functions, hamEmbed() learns median thresholds for training purpose, while binEmbedings determines hamming Embeddings based on 	medians.
* demo.py: takes test features and produces output in required format
* testHE.py and query_code.py: used for evaluation purposes. Test driver code with and without HE repectively.
