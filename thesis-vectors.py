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

'''
def l1_normalize(v):
    norm = np.sum(v)
    return v / norm
'''

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
    if (post[i,1]==post[i+1,1]) or post[i,2]>1:
      if flag==0:
        u = u + 1 
        print(post[i,1])
        musers.append(post[i,1])
      flag = flag + 1
    else:
      flag=0  

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
    t=post[i,2] # times posted
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

T1 = np.array(T1).tolist()
T2 = np.array(T2).tolist()

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

print('Length of train tensor', len(faketraintnsr))
print('Length of test tensor', len(faketesttnsr))
#It should be the same

# Compute Rescal for Fake & Real Train Tensors without the test post
print('Begin decomposing with Rescal')

rnk = 40 
print ('Rank is', rnk)

A1, R1, _, _, _ = nonneg_rescal(faketraintnsr, rnk, lambda_A=2, lambda_R=2, lambda_V=2)
print('End of 1st Rescal, computed A1')
A2, R2, _, _, _ = nonneg_rescal(realtraintnsr, rnk, lambda_A=2, lambda_R=2, lambda_V=2) 
print('End of 2nd Rescal, computed A2')

# Create feature vectors 
A1 = A1.mean(axis=0)
A2 = A2.mean(axis=0)

# Compute some values for debugging
normA1 = np.linalg.norm(A1)
normA2 = np.linalg.norm(A2)
maxA1 = np.max(A1)
maxA2 = np.max(A2)
minA1 = np.min(A1[np.nonzero(A1)])
minA2 = np.min(A2[np.nonzero(A2)])
indA1 = np.unravel_index(np.argmax(A1, axis=None), A1.shape)
indA2 = np.unravel_index(np.argmax(A2, axis=None), A2.shape)
normA12 = spatial.distance.cosine(A1,A2)

# Add test posts in Fake & Real Train Tensors, First For Fake1), np.max(A2), np.min(A1[np.nonzero(A1)]), np.min(A2[np.nonzero(A2)]))

for i in range(20):
     print('Fake Test Post:', 100+i)
     addtestpost = T1[100+i]
     
     A5, R5, _, _, _ = nonneg([addtestpost], rnk, lambda_A=2, lambda_R=2, lambda_V=2)
     
     # Create features vector 
     A5 = A5.mean(axis=0)
     
     # Compute some elements of A5
     normA5 = np.linalg.norm(A5) 
     maxA5 = np.max(A5)
     minA5 = np.min(A5)
     indA5 = np.unravel_index(np.argmax(A5, axis=None), A5.shape)
     
     # Compute Cosine distances
     normA15 = spatial.distance.cosine(A1,A5)
     normA25 = spatial.distance.cosine(A2,A5)
     
     # Remove last element in order to add the new Test post
     faketesttnsr.pop(len(faketesttnsr)-1)
     realtesttnsr.pop(len(realtesttnsr)-1)
     
     # Decide prediction 
     if normA15<normA25:
       y_pred.append(1)
       print('fake,correct')
     else:
       y_pred.append(0)
       print('real,wrong')
 
#  Same for Fake Train Tensor      
for j in range(20):
     addtestpost = T2[100+j]
     
     A5, R5, _, _, _ = nonneg([addtestpost], rnk, lambda_A=2, lambda_R=2, lambda_V=2)
     A5 = A5.mean(axis=0)
     
     normA5 = np.linalg.norm(A5)
     maxA5 = np.max(A5)
     minA5 = np.min(A5)
     indA5 = np.unravel_index(np.argmax(A5, axis=None), A5.shape)
     
     normA15 = spatial.distance.cosine(A1,A5)
     normA25 = spatial.distance.cosine(A2,A5)
     normA12 = spatial.distance.cosine(A1,A2)

     faketesttnsr.pop(len(faketesttnsr)-1)
     realtesttnsr.pop(len(realtesttnsr)-1)
     
     if normA15<normA25: 
        y_pred.append(1)
        print('fake,wrong')
     else: 
        y_pred.append(0)
        print('real,correct')

# Compute scores 
acc = accuracy_score(y_true, y_pred, normalize=False)
reslt = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary')
print('Accurate:', acc)
print('Accuracy Score:',acc/(2*r))
print('Result=',reslt)
print(y_pred)

# END
