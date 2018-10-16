import pandas as pd
import numpy as np 
from numpy import array
from scipy.sparse import *
from scipy import *
import sparse
from sktensor.rescal import als as rescal_als
from scipy.spatial import distance
from sklearn.model_selection import TimeSeriesSplit

# Create Post-User and User-User arrays
post = np.loadtxt( 'PolitiFactNewsUser.txt' )
user = np.loadtxt('PolitiFactUserUser.txt')
post = post.astype(int)
user = user.astype(int)

# Create u x u array  with the follower - followee scheme 
total = np.zeros((23865,23865), dtype=int)
for i in range(574744):
    u1 = user[i,0] - 1
    u2 = user[i,1] - 1
    # user u2 is followed by u1
    total[u2,u1] = 1 

# Number of users 
nu = 23865
# Initialize list of arrays, 120 in total, empty adjacency matrixes
faketnsr = []
realtnsr = []
for i in range(120):
    # Constructing an empty sparse matrix u x u 
    A = coo_matrix((nu, nu), dtype=np.int8).toarray()
    B = coo_matrix((nu, nu), dtype=np.int8).toarray()
    faketnsr.append(A)
    realtnsr.append(B)

# Create Fake and Real tensors from the follower-folowee scheme
k=0
# Rows of Post array
rows=32791
for i in range(rows):
    u=post[i,1] # u = User id
    p=post[i,0] # p = Post id
    if (p>120):      
        # i is followed by j 
        faketnsr[p-121][u-1][:]=total[u-1][:]
    else:
        realtnsr[p-1][u-1][:]=total[u-1][:]

# Load sorted by date fake & real posts created in mergefake.py & mergereal.py
sortedfake = np.loadtxt('sortedfakeposts.txt')
sortedfake = sortedfake.astype(int)
sortedreal = np.loadtxt('sortedrealposts.txt')
sortedreal = sortedreal.astype(int)

# Sort tensors gy date according to sortedfake & sortedreal arrays
sortedfaketnsr=[]
for i in range(120):
    sortedfaketnsr.append(faketnsr[sortedfake[i]-1])
sortedrealtnsr=[]
for i in range(120):
    sortedrealtnsr.append(realtnsr[sortedreal[i]-1])


print('Constructing 119 x u x u sparse tensor with the first 119 fakes in order to apply Rescal')
# T1 is my training set
T1 = []
for i in range(120):
    # Constructing an empty sparse matrix u x u 
     C = csr_matrix(sortedfaketnsr[i])
     T1.append(C)


print('Constructing 119 x u x u sparse tensor with the first 119  real in order to apply Rescal')
T2 = []
for i in range(120):
    # Constructing an empty sparse matrix u x u 
    D = csr_matrix(sortedrealtnsr[i])
    T2.append(D)

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
y=[]
X = sortedfaketnsr
for i in range(120):
    y.append(i)
y = np.asarray(y)
tscv = TimeSeriesSplit(n_splits=119)
print(tscv)  
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
    print("TRAIN:", train_index, "TEST:", test_index)
    if (len(train_index)>50):
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

        A1, R1, _, _, _ = rescal_als(faketraintnsr, 5, lambda_A=10, lambda_R=10) # fake only
        A2, R2, _, _, _ = rescal_als(realtraintnsr, 5, lambda_A=10, lambda_R=10) # real only 
        A3, R3, _, _, _ = rescal_als(faketesttnsr, 5, lambda_A=10, lambda_R=10) # fake-fake
        A4, R4, _, _, _ = rescal_als(realtesttnsr, 5, lambda_A=10, lambda_R=10) # real-fake
        # Flatten arrays
        Aflat = np.hstack(A1) #faketensor
        Bflat = np.hstack(A2) #realtensor
        Cflat = np.hstack(A3) #fake-fake
        Dflat = np.hstack(A4) #real-fake
        dist1 = distance.cosine(Aflat, Cflat)
        dist2 = distance.cosine(Bflat, Dflat)
        result1 = np.linalg.norm(Aflat-Cflat)
        result2 = np.linalg.norm(Bflat-Dflat)
        print("Distance between fake A1 and fake-fake A3 is:", dist1, "or", result1)
        print("Distance between real A2 and real-fake A4 is:", dist2, "or", result2)





