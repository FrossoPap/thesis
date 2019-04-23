import pandas as pd
import numpy as np 
from numpy import array
from scipy.sparse import *
from scipy import *
import sparse
from scipy.spatial import distance
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
from sklearn import svm
from sktensor import dtensor, ktensor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from classcpf import als as cp_class


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
    if (post[i,1]==post[i+1,1]) or post[i,2]>2:
      if flag==1:
        u = u + 1
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

print('Initializing list of arrays, u in total empty adjacency matrixes..')
faketnsr = []
realtnsr = []
sortedfaketnsr = []
sortedrealtnsr = []

for i in range(len(musers)):
    # Constructing an empty sparse matrix 120 x u 
    A = coo_matrix((120, u), dtype=np.int8).toarray()
    B = coo_matrix((120, u), dtype=np.int8).toarray()
    faketnsr.append(A)
    realtnsr.append(B)
    sortedfaketnsr.append(A)
    sortedrealtnsr.append(B)

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
          faketnsr[indx][p-121][:]=total[indx][:]
       else:
          realtnsr[indx][p-1][:]=total[indx][:]

# CP Decomposition
print('Merging Real and Fake Sets, 240 posts (rows) and', len(musers), 'users (slices) in total..')
# Initializing X array with Real and Fake Train Set
tr = 192
tst = (240 - tr)/2
X_tr = []    # for train set
Xnew = [] # for test set
for i in range(len(musers)):
    # Constructing an empty sparse matrix 120 x u 
    A = coo_matrix((192, len(musers)), dtype=np.int8).toarray()
    X_tr.append(A)
    B = coo_matrix((1, len(musers)), dtype=np.int8).toarray()
    Xnew.append(B)
 
# Create train set, 192 posts
for i in range(len(musers)):
   k = 0
   for j in range(0,192,2):
    X_tr[i][j][:] = faketnsr[i][k][:]
    X_tr[i][j+1][:] = realtnsr[i][k][:]
    k = k + 1  

print(X_tr[0].shape)
print('Densifying X_tr tensor..')
T1 = dtensor(X_tr)
print('Shape of tensor:', tf.shape(T1))

rnk = 5
print('Rank is:', rnk)  

print(T1.shape[0], T1.shape[1], T1.shape[2])

print('Creating label array, 1 means fake, 0 means real..')
y_train = []
for i in range(96):
   y_train.append(1)
   y_train.append(0)

y_test = []
for i in range(24):
   y_test.append(1)
for i in range(24):
   y_test.append(0)

X_test = np.zeros((48, rnk))
print('CP-CLASS decomposition for train set..')
P1, W, D, Yl, y_predtr, fit1, itr1 = cp_class(T1, y_train, rnk, init='random')
X_train = P1.U[1]

print((P1.U[0]).shape, (P1.U[1]).shape, (P1.U[2]).shape)
print('Shape of decomposed array:', X_train.shape)

from foldinf import *
print('Folding in...')

for w in range(10):
 #print('FAKE')
 for j in range(24):
   #print('Add fake test post no:', 96+j,'in line',j)	
   for i in range(len(musers)):
     Xnew[i][0][:] = faketnsr[i][96+j][:]
   Xnew = dtensor(Xnew)
   Unew, Pnew = fold_in(Xnew, W, Yl, P1.U, rnk, init='random')
   #Unew, Pnew = fold_in(Xnew, P1.U, rnk, init='random')
   #where_are_NaNs = isnan(Unew)
   #Unew[where_are_NaNs] = 0 
   #print('Unew:', Unew)
   X_test[j] = Unew
   #print('X_test:', X_test[j])
   #X_test.append(Unew)

 #print('REAL')
 for kk in range(24):
   #print('Add real test post no:', 96+k, 'in line:', 24+k)
   for i in range(len(musers)):
     Xnew[i][0][:] = realtnsr[i][96+kk][:]
   Xnew = dtensor(Xnew)
   Unew, Pnew = fold_in(Xnew, W, Yl, P1.U, rnk, init='random')
   #Unew, Pnew = fold_in(Xnew, P1.U, rnk, init='random')
   #where_are_NaNs = isnan(Unew)
   #Unew[where_are_NaNs] = 0
   X_test[24+kk] = Unew
   #X_test.append(Unew)
   #print(Unew)

 #print(X_test)

 #print('ytest:')
 #rint(y_test)

 #print('W')
 #print(W)

 #print('X_test')
 #print(X_test)

 y_pred = dot(X_test,W)

 print('ypred:, withou 0,1')
 print(y_pred)

 y_pred[abs(y_pred) > 0.3] = 1
 y_pred[abs(y_pred) < 0.3] = 0

 #print(y_pred)

 print('Results:')
 print(confusion_matrix(y_test, y_pred))
 print(classification_report(y_test, y_pred))
 acc = accuracy_score(y_test, y_pred, normalize=False)
 print('Accuracy:', acc/48)

# END

