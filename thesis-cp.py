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
from sktensor import dtensor, cp_als, ktensor

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

print('tensorflow')
import tensorflow as tf

print('start')
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
print('len of users', len(musers))
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
       
print('Loading sorted by date fake & real posts created in mergefake.py & mergereal.py..')
sortedfake = np.loadtxt('sortedfakeposts.txt')
sortedfake = sortedfake.astype(int)
sortedreal = np.loadtxt('sortedrealposts.txt')
sortedreal = sortedreal.astype(int)

print('Sorting tensors by date according to sortedfake & sortedreal arrays..')

i=0
for i in range(120):
    #print('row from initial:', sortedfake[i]-1)
    for j in range(len(musers)):
      sortedfaketnsr[j][i][:]=faketnsr[j][sortedfake[i]-1][:]
      
for i in range(120):
    #print('row from initial:', sortedreal[i]-1)
    for j in range(len(musers)):
      sortedrealtnsr[j][i][:]=realtnsr[j][sortedreal[i]-1][:]

# CP Decomposition

print('Merging Real and Fake Sets, 240 posts (rows) and', len(musers), 'users (slices) in total..')

# Initializing X array with Real and Fake Train Set
X = [] 
for i in range(len(musers)):
    # Constructing an empty sparse matrix 120 x u 
    A = coo_matrix((240, len(musers)), dtype=np.int8).toarray()
    X.append(A)

# Merging, 240 rows

for i in range(len(musers)):
   k = 0
   for j in range(0,240,2):
    X[i][j][:] = sortedfaketnsr[i][k][:]
    X[i][j+1][:] = sortedrealtnsr[i][k][:]
    k = k + 1 
   print('k:',k)

print('Densifying X tensor..')
T1 = dtensor(X)

rnk = 5
print('Rank is:', rnk)  

print('CP decomposition for tensor..')
P1, fit1, itr1, exectimes1 = cp_als(T1, rnk, init='random')

# We need mode 2 Decomposition
X = P1.U[1]
print('Shape of decomposed array:', X.shape)

print('Creating label array, 1 means fake, 0 means real..')
y = []
for i in range(120):
   y.append(1)
   y.append(0)

print('Number of labels:', length(y))

# X holds the feature matrix (240 x rnk) 

print('Creating Train and Test Sets..')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 

print('Fitting the model..(Sigmoid SVC Kernel)')
svclassifier = SVC(kernel='sigmoid')  
svclassifier.fit(X_train, y_train)  
y_pred = svclassifier.predict(X_test)  

print('Results:')
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

print('Fitting the model..(Gaussian SVC Kernel)')
svclassifier = SVC(kernel='rbf')  
svclassifier.fit(X_train, y_train) 
y_pred = svclassifier.predict(X_test)  

print('Results:')
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

# END

