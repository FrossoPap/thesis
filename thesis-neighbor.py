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
'''
for i in range(32790):
    if (post[i,1]==post[i+1,1]):
      if flag==0:
        u = u + 1 
        print(post[i,1])
        musers.append(post[i,1])
      flag = flag + 1
    else:
      flag=0  
print(musers)
'''

for i in range(32790):
    if (post[i,1]==post[i+1,1]):
      if post[i,1] not in musers:
        u = u + 1
        print(post[i,1])
        musers.append(post[i,1])
      flag = flag + 1
    else:
      flag=0

print(u)

#print(musers[4140])



'''
for i in range(23865):
    musers.append(i+1)

u = 23865
'''
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

# Initialize true & prediction labels
y_true = []
y_pred = []

# Labels for True Test Set, 0 means it is true
for i in range(20):
    y_true.append(0)

# Labels for Fake Test Set, 1 means it is fake
for i in range(20):
    y_true.append(1)

t = 100
simil = 0
flag = 0
for j in range(100,120):
  print('REAL POST', j)
  simil = 0 
  flag = 0 
  B = sortedrealtnsr[j]
  print(np.sum(B!=0))
  B = B.flatten()
  for i in range(t):
    #print('FAKE TRAIN SET POST', i)
    A = sortedfaketnsr[i]
    A = A.flatten()
    s = np.sum((A==B) & B!=0)
    if s!=0: print(s)
    test = 1 - spatial.distance.cosine(A,B)
    #test = np.linalg.norm(A-B)
    #print(simil)
    if test>simil:
       simil = test
       flag = 0
  print(simil) 

  #print('REAL')
  for k in range(t):
    #print('REAL TRAIN SET POST', k)
    A = sortedrealtnsr[k]
    A = A.flatten()
    s= np.sum((A==B) & B!=0)
    if s!=0: print(s)
    test = 1 - spatial.distance.cosine(A,B)
    #test = np.linalg.norm(A-B)
    if test>simil:
       simil = test 
       flag = 1
  print(simil)

  if flag==0:
    print('fake, wrong')
    print(simil)
    y_pred.append(1)
  else:
    print('real, correct')
    print(simil)
    y_pred.append(0)

simil = 0
flag = 0
j = 0
for j in range(100,120):
  print('FAKE POST', j)
  simil = 0 
  flag = 0
  B = sortedfaketnsr[j]
  print(np.sum(B!=0))
  B = B.flatten()
  for i in range(t):
    #print('FAKE TRAIN SET POST:', i)
    A = sortedfaketnsr[i]
    A = A.flatten()
    s = np.sum((A==B) & B!=0)
    if s!=0: print(s)
    test = 1 - spatial.distance.cosine(A,B)
    #print(simil)
    if test>simil:
       simil = test
       flag = 0
  print(simil) 
  #print('REAL')
  for k in range(t):
    #print('REAL TRAIN SET POST:', k)
    A = sortedrealtnsr[k]
    A = A.flatten()
    test = 1 - spatial.distance.cosine(A,B)
    s = np.sum((A==B) & B!=0)
    if s!=0: print(s)
    if test>simil:
       simil = test
       flag = 1
  print(simil)
  
  if flag==0:
    print('fake, correct')
    y_pred.append(1)
    print(simil)
  else:
    print('real, wrong')
    y_pred.append(0)
    print(simil)

acc = accuracy_score(y_true, y_pred, normalize=False)
reslt = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary')
print('Accurate:', acc)
print('Accuracy Score:',acc/(2*20))
print('Result=',reslt)
print(y_pred)


