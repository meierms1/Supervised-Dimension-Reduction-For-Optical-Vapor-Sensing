#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:16:05 2021

@author: maycon
"""

from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import scipy
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA,KernelPCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
warnings.filterwarnings('ignore')
from Methods import SPCA,RPCA, my_train_test_split, opt_param
from sklearn.metrics import zero_one_loss, accuracy_score, precision_score


testFraction = 0.25 # used to split the data
numberOfComponents = 2 #amount of components after dimension reduction
numberOfComponents2 = 2
method = 2 #Method 1 = sklearn split // Method 2 = Custom split
sheet_name = 6 #Choose the tab you want to read from the excel sheet [0 to 5]
split_control = 3 # number of test point per chemical, either 1, 2, 3, or 4.
loop_size = 1 #amount of trys for each method
local = '../mplot' + str(sheet_name + 1) + str(method)+ '/' #folder to save graphs
split_seed = None

path =  r"/Users/maycon/Desktop/Codes_Properties/Differential Reflectance Data.xlsx"

df = pd.read_excel(path, sheet_name=sheet_name)
spectraData = df.values

X = spectraData[:,2:spectraData.shape[1]]

range_control = 15

speciesCount = 0;
Y = np.empty([X.shape[1], 1])
for columnCount in range(0, X.shape[1]):
    i = np.remainder(columnCount, range_control)
    if (i < range_control):
        Y[columnCount] = speciesCount
    if (i == range_control - 1):
        speciesCount += 1

track = np.array([[0.02],[0.05],[0.1],[0.2],[0.3],
         [0.02],[0.05],[0.1],[0.2],[0.3],
         [0.02],[0.05],[0.1],[0.2],[0.3],
         
         [0.02],[0.05],[0.1],[0.2],[0.3],
         [0.02],[0.05],[0.1],[0.2],[0.3],
         [0.02],[0.05],[0.1],[0.2],[0.3],
         
         [0.02],[0.05],[0.1],[0.2],[0.3],
         [0.02],[0.05],[0.1],[0.2],[0.3],
         [0.02],[0.05],[0.1],[0.2],[0.3],
         
         [0.02],[0.05],[0.1],[0.2],[0.3],
         [0.02],[0.05],[0.1],[0.2],[0.3],
         [0.02],[0.05],[0.1],[0.2],[0.3],
         
         [0.02],[0.05],[0.1],[0.2],[0.3],
         [0.02],[0.05],[0.1],[0.2],[0.3],
         [0.02],[0.05],[0.1],[0.2],[0.3],
         ])
        
X = np.transpose(X)
X = StandardScaler().fit_transform(X)      



''' 
LDA Implement
'''
seed_l = 296
#max = 1
XTrain, XTest, YTrain1, YTest1 = my_train_test_split(X, Y, test_size=split_control, range_control = range_control, seed = seed_l)
tr1, tr2,_,_ = my_train_test_split(track,track, test_size=split_control, range_control = range_control, seed = seed_l)
track_lda = np.concatenate((tr1,tr2))
numberOfTrainingPoints = XTrain.shape[0]
totalDataPoints = X.shape[0]
XNew = np.concatenate((XTrain, XTest)) 

pcomp = 30         

pca = PCA(n_components=pcomp)
XReduced = pca.fit_transform(XNew)
            
XReducedTrain = XReduced[0:numberOfTrainingPoints,:];
XReducedTest = XReduced[numberOfTrainingPoints : totalDataPoints,:];
            
lda = LinearDiscriminantAnalysis(n_components = numberOfComponents2, priors
                                     = None, shrinkage = None, solver = 'eigen')
lda.fit(XReducedTrain,YTrain1)
XTrainLDA = lda.transform(XReducedTrain)
XTestLDA = lda.transform(XReducedTest)



    
''' 
pls 
''' 
seed_pls = 25
#max = 0.867
XTrain, XTest, YTrain2, YTest2 = my_train_test_split(X, Y, test_size=split_control, range_control = range_control, seed = seed_pls)
tr1, tr2,_,_  = my_train_test_split(track,track, test_size=split_control, range_control = range_control, seed = seed_pls)
track_pls = np.concatenate((tr1,tr2))
pls = PLSRegression(n_components=numberOfComponents)
pls.fit(XTrain, YTrain2)
XTrainPLS = pls.transform(XTrain)
XTestPLS = pls.transform(XTest)


    
''' 
pca 
'''
seed_p = 29
#max 0.73
pca = PCA(n_components=numberOfComponents)
principalComponents = pca.fit_transform(X)
    
''' 
spca 
'''
seed_s = 18
#max 0.8
XTrain, XTest, YTrain3, YTest3 = my_train_test_split(X, Y, test_size=split_control, range_control = range_control, seed = seed_s)
tr1, tr2,_,_  = my_train_test_split(track,track, test_size=split_control, range_control = range_control, seed = seed_s)
track_spca = np.concatenate((tr1,tr2))
numberOfTrainingPoints = XTrain.shape[0]
L = np.zeros((numberOfTrainingPoints, numberOfTrainingPoints), dtype=np.int8)
for data_index_row in range(numberOfTrainingPoints):
    for data_index_column in range(numberOfTrainingPoints):
        L[data_index_row, data_index_column] = np.equal(YTrain3[data_index_row],YTrain3[data_index_column])
    
L= L + np.eye(numberOfTrainingPoints, dtype = np.int8)
[XTrainSPCA, XTestSPCA] = SPCA(XTrain, XTest, L, numberOfComponents)
###################################################

    
''' 
rpca 
'''
seed_r = 296
# max 1
XTrain, XTest, YTrain4, YTest4 = my_train_test_split(X, Y, test_size=split_control, range_control = range_control, seed = seed_r)
tr1, tr2,_,_  = my_train_test_split(track,track, test_size=split_control, range_control = range_control, seed = seed_r)
track_rpca = np.concatenate((tr1,tr2))
totalDataPoints = X.shape[0]
XNew = np.concatenate((XTrain,XTest))
pca = PCA(n_components = pcomp)
XReduced = pca.fit_transform(XNew)
XReducedTrain = XReduced[0:numberOfTrainingPoints,:];
XReducedTest = XReduced[numberOfTrainingPoints : totalDataPoints,:];
[XTrainRPCA, XTestRPCA] = RPCA(XReducedTrain, XReducedTest, L, numberOfComponents)


###############################################################


    
''' 
kpca
'''
seed_k = 145
#max 0.66
kpca = KernelPCA(n_components=numberOfComponents, kernel='rbf',
                     gamma=None, eigen_solver = 'arpack', max_iter = 6000)
XKPCA = kpca.fit_transform(X) 

        
###################################################


to_save = {'Y': Y,'XKPCA': XKPCA, 'XPCA': principalComponents, 
           'XTrainLDA':XTrainLDA, 'XTestLDA':XTestLDA, 'YTrainLDA':YTrain1, 'YTestLDA':YTest1,
           'XTrainRPCA':XTrainRPCA, 'XTestRPCA':XTestRPCA, 'YTrainRPCA':YTrain4, 'YTestRPCA':YTest4,
           'XTrainSPCA':XTrainSPCA, 'XTestSPCA':XTestSPCA, 'YTrainSPCA':YTrain3, 'YTestSPCA':YTest3,
           'XTrainPLS':XTrainPLS, 'XTestPLS':XTestPLS, 'YTrainPLS':YTrain2, 'YTestPLS':YTest2,
           "trackp": track_pls , "trackl": track_lda , "tracks": track_spca, "trackr": track_rpca, "trackpca":track}

savemat('all_data.mat', to_save)

