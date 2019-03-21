# coding: utf-8
"""
This module holds algorithms to compute the folding-in of CP decomposition
P.U[0], P.U[2] remain unchanged
Only P.U[1] changes (Shape: train posts x users)
"""
import logging
import time
import numpy as np
from numpy import array, dot, ones, sqrt
from scipy.linalg import pinv, inv
from numpy.random import rand
from sktensor import * 
from sktensor.core import nvecs, norm
from sktensor.ktensor import ktensor
import tensorflow as tf

_log = logging.getLogger('CP')
_DEF_MAXITER = 50
_DEF_INIT = 'nvecs'
_DEF_CONV = 1e-5
_DEF_FIT_METHOD = 'full'
_DEF_TYPE = np.float

__all__ = [
    'fold_in',
    'opt',
    'wopt'
]


def fold_in(X, Uold, rank, **kwargs):

    # init options
    ainit = kwargs.pop('init', _DEF_INIT)
    maxiter = kwargs.pop('max_iter', _DEF_MAXITER)
    fit_method = kwargs.pop('fit_method', _DEF_FIT_METHOD)
    conv = kwargs.pop('conv', _DEF_CONV)
    dtype = kwargs.pop('dtype', _DEF_TYPE)
    
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    N = X.ndim
    normX = norm(X)

    U = _init(ainit, X, N, Uold, rank, dtype)
    fit = 0
    exectimes = []

    for itr in range(maxiter):
        tic = time.clock()
        fitold = fit
        n = 1 # Mode 1 is the array that changes
        Unew = X.uttkrp(U, n)
        '''
        # Can't implement because of memory error
        n, p = U[0].shape
        m, pC = U[2].shape
        C = np.einsum('ij, kj -> ikj', U[0], U[2]).reshape(m * n, p)
        nk, ni, nj = X.shape
        jk = nk * nj
        sess = tf.Session()
        with sess.as_default():
           Xnew = tf.reshape(X, [1,jk])
           Xnew = Xnew.eval()
        Z = Xnew.dot(pinv(C))
        Unew = (Unew.dot(Z)).dot(inv(Unew))
        ''' 
        Y = ones((rank, rank), dtype=dtype)
        for i in (list(range(n)) + list(range(n + 1, N))):
            Y = Y * dot(U[i].T, U[i])
        Unew = Unew.dot(pinv(Y))
        
        # Normalize
        if itr == 0:
            lmbda = sqrt((Unew ** 2).sum(axis=0))
        else:
            lmbda = Unew.max(axis=0)
            lmbda[lmbda < 1] = 1
        
        U[1] = Unew / lmbda
        P = ktensor(U, lmbda)

        if fit_method == 'full':
            normresidual = normX ** 2 + P.norm() ** 2 - 2 * P.innerprod(X)
            #normresidual = normX ** 2 + np.linalg.norm(U[1]) ** 2 - 2 * (np.linalg.norm(U[1]))*(X)
            fit = 1 - (normresidual / normX ** 2)
        else:
            fit = itr
        
        fitchange = abs(fitold - fit)
        exectimes.append(time.clock() - tic)
        
        if itr > 0 and fitchange < conv:
            break

    return U[1], P


def opt(X, rank, **kwargs):
    ainit = kwargs.pop('init', _DEF_INIT)
    maxiter = kwargs.pop('maxIter', _DEF_MAXITER)
    conv = kwargs.pop('conv', _DEF_CONV)
    dtype = kwargs.pop('dtype', _DEF_TYPE)
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    N = X.ndim
    U = _init(ainit, X, N, rank, dtype)


def wopt(X, rank, **kwargs):
    raise NotImplementedError()


def _init(init, X, N, Uold, rank, dtype):
    """
    Initialization for CP models
    Only Uinit[1] changes, rest remain unchanged

    """
    Uinit = [None for _ in range(N)]
    Uinit[0] = Uold[0]
    #Uinit[1] = Uold[1] 
    Uinit[2] = Uold[2]
        # Or initialize Uinit[1] from start (?)   
    n = 1 
    if isinstance(init, list):
        Uinit = init
    elif init == 'random':
        Uinit[1] = array(rand(X.shape[n], rank), dtype=dtype)
    elif init == 'nvecs':
        Uinit[1] = array(nvecs(X, n, rank), dtype=dtype)
    else:
        raise 'Unknown option (init=%s)' % str(init)
    return Uinit
     
