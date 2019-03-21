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
import tensorflow as tf


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
X = [] 
Xnew = []
for i in range(len(musers)):
    # Constructing an empty sparse matrix 120 x u 
    A = coo_matrix((192, len(musers)), dtype=np.int8).toarray()
    X.append(A)
    B = coo_matrix((1, len(musers)), dtype=np.int8).toarray()
    Xnew.append(B)
 
# Merging, 240 rows
for i in range(len(musers)):
   k = 0
   for j in range(0,192,2):
    X[i][j][:] = faketnsr[i][k][:]
    X[i][j+1][:] = realtnsr[i][k][:]
    k = k + 1 


print('K is:', k)
print(X[0].shape)
print('Densifying X tensor..')
T1 = dtensor(X)
print('Shape of tensor:', tf.shape(T1))

rnk = 15 
print('Rank is:', rnk)  
print(T1.shape[0], T1.shape[1], T1.shape[2])
print('CP decomposition for tensor..')
P1, fit1, itr1, exectimes1 = cp_als(T1, rnk, init='random')

print('End of 1st Decomposition')
X_train = P1.U[1]

print((P1.U[0]).shape, (P1.U[1]).shape, (P1.U[2]).shape)
print('Shape of decomposed array:', X_train.shape)

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

from foldin2 import *

X_test = coo_matrix((48, rnk), dtype=np.int8).toarray()

print('Folding in...')

for j in range(24):
  print('Add fake test post no:', 96+j,'in line',j)	
  for i in range(len(musers)):
    Xnew[i][0][:] = faketnsr[i][96+j][:]
  Xnew = dtensor(Xnew)
  Unew, Pnew = fold_in(Xnew, P1.U, rnk, init='random') 
  X_test[j] = Unew

for k in range(24):
  print('Add real test post no:', 96+k, 'in line:', 24+k)
  for i in range(len(musers)):
    Xnew[i][0][:] = realtnsr[i][96+k][:]
  Xnew = dtensor(Xnew)
  Unew, Pnew = fold_in(Xnew, P1.U, rnk, init='random')
  X_test[24+k] = Unew

'''
print('Number of labels:', len(y))
# X holds the feature matrix (240 x rnk) 
print('Creating Train and Test Sets..')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42) 
'''

print('Y_test:', y_test)
print('Fitting the model..(Sigmoid SVM Kernel)')
svclassifier = SVC(kernel='sigmoid')  
svclassifier.fit(X_train, y_train)  
y_pred = svclassifier.predict(X_test)  

print('Results:')
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

print('Fitting the model..(Gaussian SVM Kernel)')
svclassifier = SVC(kernel='rbf')  
svclassifier.fit(X_train, y_train) 
y_pred = svclassifier.predict(X_test)  

print('Results:')
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


print('Fitting the model..(LINEAR SVM Kernel)')
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

print('Results:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier  

print('Fitting the model..(KNN)')
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)  

print('Results:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# END

