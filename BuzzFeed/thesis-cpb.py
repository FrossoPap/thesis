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
#from sktensor import dtensor, cp_als, ktensor
from sktensor import dtensor, ktensor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from cpb import als as cp_als

print('Creating Post-User and User-User arrays..')
post = np.loadtxt('BuzzFeedNewsUser.txt' )
user = np.loadtxt('BuzzFeedUserUser.txt')
post = post.astype(int)
user = user.astype(int)

print('Counting number of users with more than one interaction..')
flag = 0
u = 0 
musers = []
for i in range(22778):
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
for i in range(634750):
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
    A = coo_matrix((91, u), dtype=np.int8).toarray()
    B = coo_matrix((91, u), dtype=np.int8).toarray()
    faketnsr.append(A)
    realtnsr.append(B)
    sortedfaketnsr.append(A)
    sortedrealtnsr.append(B)

print('Creating Fake and Real tensors from the follower-folowee scheme..')
# Rows of Post array
rows = 22778
for i in range(rows):
    u=post[i,1] # u = User id
    p=post[i,0] # p = Post id
    if u in musers:
       indx = musers.index(u)
       # Post ids larger than 120 are Fake
       if (p>91):      
          # i is followed by j
          # Copy following scheme 
          faketnsr[indx][p-92][:]=total[indx][:]
       else:
          realtnsr[indx][p-1][:]=total[indx][:]


# CP Decomposition
print('Merging Real and Fake Sets, 182 posts (rows) and', len(musers), 'users (slices) in total..')
# Initializing X array with Real and Fake Train Set
X = [] 
for i in range(len(musers)):
    # Constructing an empty sparse matrix 120 x u 
    A = coo_matrix((182, len(musers)), dtype=np.int8).toarray()
    X.append(A)

# Merging, 240 rows
for i in range(len(musers)):
   k = 0
   for j in range(0,182,2):
    X[i][j][:] = faketnsr[i][k][:]
    X[i][j+1][:] = realtnsr[i][k][:]
    k = k + 1 

print(X[0].shape)
print('Densifying X tensor..')
T1 = dtensor(X)
print('Shape of tensor:', tf.shape(T1))

rnk = 10 
print('Rank is:', rnk)  
print(T1.shape[0], T1.shape[1], T1.shape[2])
print('CP decomposition for tensor..')

print('Creating label array, 1 means fake, 0 means real..')
y = []
for i in range(91):
   y.append(1)
   y.append(0)

for i in range(10):
   P1, fit1, itr1, exectimes1 = cp_als(T1, rnk, init='random')
   X = P1.U[1]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, shuffle=False)

   print('LINEAR SVM Kernel')
   svclassifier = SVC(kernel='linear')
   svclassifier.fit(X_train, y_train)
   y_pred = svclassifier.predict(X_test)

   print('Results:')
   print(confusion_matrix(y_test, y_pred))
   print(classification_report(y_test, y_pred))
   acc = accuracy_score(y_test, y_pred, normalize=False)
   print('Accuracy:', acc)

   from sklearn.neighbors import KNeighborsClassifier  

   print('KNN')
   classifier = KNeighborsClassifier(n_neighbors=5)  
   classifier.fit(X_train, y_train)  
   y_pred = classifier.predict(X_test)  

   print('Results:')
   print(confusion_matrix(y_test, y_pred))
   print(classification_report(y_test, y_pred))
   acc = accuracy_score(y_test, y_pred, normalize=False)
   print('Accuracy:', acc)




# END

