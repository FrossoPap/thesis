import pandas as pd
import numpy as np 
from numpy import array
from scipy.sparse import *
from scipy import *
import sparse
from sktensor.rescal import als as rescal_als
from scipy.spatial import distance
from sklearn.model_selection import TimeSeriesSplit
import logging
import time
import numpy as np
from numpy import dot
from numpy import array
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import issparse
from numpy.random import rand
from sklearn.decomposition.nmf import _initialize_nmf

T1 = np.load('T1.npy')
T2 = np.load('T2.npy')

T1 = np.array(T1).tolist()
T2 = np.array(T2).tolist()

k1=0
k2=0
result=[]
for j in range(100):
    print('Iteration:',j)
    X = sortedfaketnsr
    tscv = TimeSeriesSplit(n_splits=119)
#     print(tscv)  
    TimeSeriesSplit(max_train_size=None, n_splits=119)
    for train_index, test_index in tscv.split(X):
        R1 = []
        R2 = []
        R3 = []
        R4 = []
        A1 = []
        A1 = np.asarray(A1)
        A2 = []
        A2 = np.asarray(A2)
        A3 = []
        A3 = np.asarray(A3)
        A4 = []
        A4 = np.asarray(A4)
        faketraintnsr=[]
        faketesttnsr=[]
        realtraintnsr=[]
        realtesttnsr=[]
        addftrainpost = []
        addtrtrainpost = []
        dist1 = 0
        dist2 = 0
#        print("TRAIN:", train_index, "TEST:", test_index)
        if len(train_index)==10:
            for i in range(len(train_index)):
                    addftrainpost = T1[train_index[i]]
                    addrtrainpost = T2[train_index[i]]
                    faketraintnsr.append(addftrainpost)
                    realtraintnsr.append(addrtrainpost)
                    faketesttnsr.append(addftrainpost)
                    realtesttnsr.append(addrtrainpost)
            addtestpost = T1[test_index[0]]
            faketesttnsr.append(addtestpost)
            realtesttnsr.append(addtestpost)
            A1, R1, _, _, _ = nonneg_rescal(faketraintnsr, 10, lambda_A=0.01, lambda_R=0.01) # fake only
            print('end of 1st Rescal')
            A2, R2, _, _, _ = nonneg_rescal(realtraintnsr, 10, lambda_A=0.01, lambda_R=0.01) # real only
            print('end of 2nd Rescal')
            A3, R3, _, _, _ = nonneg_rescal(faketesttnsr, 10, lambda_A=0.01, lambda_R=0.01) # fake-fake
            print('end of 3rd Rescal')
            A4, R4, _, _, _ = nonneg_rescal(realtesttnsr, 10, lambda_A=0.01, lambda_R=0.01) # real-fake
            print('end of 4th Rescal')
	    # Flatten arrays
            Aflat = np.hstack(A1) #faketensor
            Bflat = np.hstack(A2) #realtensor
            Cflat = np.hstack(A3) #fake-fake
            Dflat = np.hstack(A4) #real-fake
#            dist1 = distance.cosine(Aflat, Cflat)
#            dist2 = distance.cosine(Bflat, Dflat)
            result1 = np.linalg.norm(A1-A3)
            result2 = np.linalg.norm(A2-A4)
            print("Distance between fake A1 and fake-fake A3 is:", result1, "in norm")
            print("Distance between real A2 and real-fake A4 is:", result2)
            if result1<result2:
                print('Prediction was correct')
                k1 = k1 + 1 
            else:
                print('Prediction was wrong')
           
print('Prediction accuracy is', k1, '%')
# END
