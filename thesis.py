import pandas as pd
import numpy as np 
from numpy import array
from scipy.sparse import *
from scipy import *
import sparse
from sktensor.rescal import als as rescal_als
from scipy.spatial import distance

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
sortedfake = sortedfakes.astype(int)
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
for i in range(119):
    # Constructing an empty sparse matrix u x u 
     C = csr_matrix(sortedfaketnsr[i])
     T1.append(C)

print('Rescal on 119 first fakes')
A1, R1, _, _, _ = rescal_als(T1, 2)

print('Save Results, A1,T1...')
np.savetxt('A1.txt', A1)
np.save('T1.txt', T1)

print('Constructing 119 x u x u sparse tensor with the first 119  real in order to apply Rescal')
T2 = []
for i in range(119):
    # Constructing an empty sparse matrix u x u 
     D = csr_matrix(sortedrealtnsr[i])
     T2.append(D)

print('Rescal on 119 first real posts')
A2, R2, _, _, _ = rescal_als(T2, 2)

print('Constructing u x u sparse matrix for the last fake post in order to apply Rescal')
T3 = []
Z = csr_matrix(sortedfaketnsr[119])
T3.append(Z)

print('Rescal on 120th fake post')
A3, R3, _, _, _ = rescal_als(T3, 2)

# Flatten A arrays
Aflat = np.hstack(A1) 
Bflat = np.hstack(A2)
Cflat = np.hstack(A3)
dist = distance.cosine(Aflat, Cflat)
dist2 = distance.cosine(Bflat, Cflat)
result2 = np.linalg.norm(Aflat-Cflat)
result3 = np.linalg.norm(Bflat-Cflat)
print(dist)
print(dist2)
print(result3)
print(result2)






