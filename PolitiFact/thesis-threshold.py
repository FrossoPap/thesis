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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from nonneg_rescal import *
from nonneg import *
from scipy import spatial

def l1_normalize(v):
    norm = np.sum(v)
    return v / norm


print('Creating Post-User and User-User arrays..')
post = np.loadtxt('PolitiFactNewsUser.txt' )
user = np.loadtxt('PolitiFactUserUser.txt')
post = post.astype(int)
user = user.astype(int)

print('Counting number of users with more than one interaction..')
flag = 0
u = 0 
musers = []

for i in range(32790):
    if (post[i,1]==post[i+1,1]):
      if flag==0:
        u = u + 1
        musers.append(post[i,1])
      flag = flag + 1
    else:
      flag=0  
print(musers)

#for i in range(23865):
 #   musers.append(i+1)

#u = 23865

print('Number of final users is:', u)

print('Create', u,'x', u,'array with the follower-followee scheme..') 
total = np.zeros((u,u), dtype=int)
for i in range(574744):
    u1 = user[i,0]
    u2 = user[i,1]
    if u2 in musers:
      if u1 in musers:
        indx1 = musers.index(u1)
        indx2 = musers.index(u2)
        # user u2 is followed by u1
        total[indx2,indx1] = 1 

print('Initializing list of arrays, 120 in total empty adjacency matrixes..')
faketnsr = []
realtnsr = []
for i in range(120):
    # Constructing an empty sparse matrix u x u 
    A = coo_matrix((u, u), dtype=np.int8).toarray()
    B = coo_matrix((u, u), dtype=np.int8).toarray()
    faketnsr.append(A)
    realtnsr.append(B)

print('Creating Fake and Real tensors from the follower-folowee scheme..')
# Rows of Post array
rows = 32791
for i in range(rows):
    u=post[i,1] # u = User id
    p=post[i,0] # p = Post id
    if u in musers:
       indx = musers.index(u)
       # Post ids larger than 120 are Fake
       if (p>120):      
          # i is followed by j
          # Copy following scheme 
          faketnsr[p-121][indx][:]=total[indx][:]
       else:
          realtnsr[p-1][indx][:]=total[indx][:]
       
print('Loading sorted by date fake & real posts created in mergefake.py & mergereal.py..')
sortedfake = np.loadtxt('sortedfakeposts.txt')
sortedfake = sortedfake.astype(int)
sortedreal = np.loadtxt('sortedrealposts.txt')
sortedreal = sortedreal.astype(int)

print('Sorting tensors by date according to sortedfake & sortedreal arrays..')
sortedfaketnsr=[]
for i in range(120):
    sortedfaketnsr.append(faketnsr[sortedfake[i]-1])

sortedrealtnsr=[]
for i in range(120):
    sortedrealtnsr.append(realtnsr[sortedreal[i]-1])

np.save('sortedfaketnsr',sortedfaketnsr)
np.save('sortedrealtnsr',sortedrealtnsr)

#sortedfaketnsr = np.load('sortedfaketnsr.npy')
#sortedrealtnsr = np.load('sortedrealtnsr.npy')

print('Constructing 119 x u x u sparse tensor with the first 119 fakes in order to apply Rescal..')
# T1 is for the fake training set
T1 = []
for i in range(120):
    # Constructing an empty sparse matrix u x u 
     C = csr_matrix(sortedfaketnsr[i])
     T1.append(C)

print('Constructing 119 x u x u sparse tensor with the first 119 reals in order to apply Rescal..')
T2 = []
for i in range(120):
    # Constructing an empty sparse matrix u x u 
   D = csr_matrix(sortedrealtnsr[i])
   T2.append(D)

#print('Loading results..')
#np.save('T1', T1)
#np.save('T2', T2)
#T1 = np.load('T1.npy')
#T2 = np.load('T2.npy')

T1 = np.array(T1).tolist()
T2 = np.array(T2).tolist()
#print(T1)
#print(T2)

print('Begin decomposing with Rescal..')
# Initialize As and Rs for Rescal
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

# Here you change how many posts to have in the train set 
s = 100
# r is the remaining number of posts as a test set   
#r = 50
r = 20
print('My Train Set is of length ', s)
print('My Test Set is of length ', r)

# Initialize true & prediction labels
y_true = []
y_pred = []

# Labels for True Test Set, 0 means it is true
for i in range(20):
    y_true.append(1)

# Labels for Fake Test Set, 1 means it is fake
for i in range(20):
    y_true.append(0)

print('y_true=', y_true)

for i in range(s):
    addftrainpost = T1[i]
    addrtrainpost = T2[i]
    # Create Fake & Real Train Tensors up to s
    faketraintnsr.append(addftrainpost)
    realtraintnsr.append(addrtrainpost)
    # At the same time, create Fake & Real Tensors in which we will then add the test post
    faketesttnsr.append(addftrainpost)
    realtesttnsr.append(addrtrainpost)

#print('Length of train tensor', len(faketraintnsr))
#print('Length of test tensor', len(faketesttnsr))

# Compute Rescal for Fake & Real Train Tensors without the test post
rnk = 10
print ('Rank is', rnk)
A1, R1, _, _, _ = nonneg_rescal(faketraintnsr, rnk, lambda_A=2, lambda_R=2, lambda_V=2)
print('End of 1st Rescal, computed A1')
A2, R2, _, _, _ = nonneg_rescal(realtraintnsr, rnk, lambda_A=2, lambda_R=2, lambda_V=2) 
print('End of 2nd Rescal, computed A2')
#print('Zeros in A1:', np.sum(~A1.any(1)))
#print('Zeros in A2:', np.sum(~A2.any(1)))
#indexes = np.where(~A1.any(axis=1))[0]
#indexes2 = np.where(~A2.any(axis=1))[0]
#A1 = np.delete(A1, indexes, axis=0)
#A2 = np.delete(A2, indexes2, axis=0)
#print(A1)
normA1 = np.linalg.norm(A1)
normA2 = np.linalg.norm(A2)
print('Norm A1:', normA1)
print('Norm A2:', normA2)

print('np.linalg.norm A1',normA1)
print('np.linalg.norm A2',normA2)


for i in range(20):
    print('_____________________________________________________________________________________')
    print('Fake Test Post:', 100+i)
    addtestpost = T1[100+i]
    faketesttnsr.append(addtestpost)
    realtesttnsr.append(addtestpost)
    print('Fake test tnsr len:', len(faketesttnsr))
    print('Fake train tnsr len:', len(faketraintnsr))
    A3, R3, _, _, _ = nonneg(faketesttnsr, rnk, lambda_A=2, lambda_R=2, lambda_V=2)
    A4, R4, _, _, _ = nonneg(realtesttnsr, rnk, lambda_A=2, lambda_R=2, lambda_V=2)
    t = 0.99
    
    A1 = np.where(A1>t, 1, 0)
    A2 = np.where(A2>t, 1, 0)
    A3 = np.where(A3>t, 1, 0)
    A4 = np.where(A4>t, 1, 0)

    A1 = A1.flatten()
    A2 = A2.flatten()
    A3 = A3.flatten()
    A4 = A4.flatten()
    
    z1 = 1 - spatial.distance.cosine(A1,A3)
    z2 = 1 - spatial.distance.cosine(A2,A4)
    
    if z1>z2:
      print('fake',z1,z2)
      y_pred.append(1)
    else:
      print('real',z1,z2)
      y_pred.append(0)

    faketesttnsr.pop(len(faketesttnsr)-1)
    realtesttnsr.pop(len(realtesttnsr)-1)

for i in range(20):
    print('_____________________________________________________________________________________')
    print('Real Test Post:', 100+i)
    addtestpost = T2[100+i]
    faketesttnsr.append(addtestpost)
    realtesttnsr.append(addtestpost)
    print('Fake test tnsr len:', len(faketesttnsr))
    print('Fake train tnsr len:', len(faketraintnsr))
    A3, R3, _, _, _ = nonneg(faketesttnsr, rnk, lambda_A=2, lambda_R=2, lambda_V=2)
    A4, R4, _, _, _ = nonneg(realtesttnsr, rnk, lambda_A=2, lambda_R=2, lambda_V=2)
    t = 0.99

    A1 = np.where(A1>t, 1, 0)
    A2 = np.where(A2>t, 1, 0)
    A3 = np.where(A3>t, 1, 0)
    A4 = np.where(A4>t, 1, 0)

    A1 = A1.flatten()
    A2 = A2.flatten()
    A3 = A3.flatten()
    A4 = A4.flatten()

    z1 = 1 - spatial.distance.cosine(A1,A3)
    z2 = 1 - spatial.distance.cosine(A2,A4)

    if z1>z2:
      print('fake,wrong',z1,z2)
      y_pred.append(1)
    else:
      print('real,correct',z1,z2)
      y_pred.append(0)

    faketesttnsr.pop(len(faketesttnsr)-1)
    realtesttnsr.pop(len(realtesttnsr)-1)

acc = accuracy_score(y_true, y_pred, normalize=False)
reslt = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary')
print('Accurate:', acc)
print('Accuracy Score:',acc/(2*r))
print('Result=',reslt)
print(y_pred)


# END
