# coding: utf-8
# Copyright (C) 2013 Maximilian Nickel <mnick@mit.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
This module holds diffent algorithms to compute the CP decomposition, i.e.
algorithms where

.. math:: \\ten{X} \\approx \sum_{r=1}^{rank} \\vec{u}_r^{(1)} \outer \cdots \outer \\vec{u}_r^{(N)}

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

_log = logging.getLogger('CP')
_DEF_MAXITER = 500
_DEF_INIT = 'nvecs'
_DEF_CONV = 1e-5
_DEF_FIT_METHOD = 'full'
_DEF_TYPE = np.float

__all__ = [
    'als',
    'opt',
    'wopt'
]


def als(X, Yl, rank, **kwargs):
    """
    Alternating least-sqaures algorithm to compute the CP decomposition taking into 
    consideration the labels of the set
    """

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
    Yl = np.asarray(Yl)
    Yl = np.reshape(Yl, (-1,1))
    normYl = np.linalg.norm(Yl)
    U = _init(ainit, X, N, rank, dtype)
    fit = 0
    exectimes = []

    # Initialize W 
    W = ones((rank,1), dtype=dtype)
    
    for itr in range(maxiter):
        tic = time.clock()
        fitold = fit
        
        for n in range(N):
            Unew = X.uttkrp(U, n)
            Y = ones((rank, rank), dtype=dtype)
            for i in (list(range(n)) + list(range(n + 1, N))):
                Y = Y * dot(U[i].T, U[i])
            if n!=1:
                # Updates remain the same for U0,U2
                Unew = Unew.dot(pinv(Y))
            else:
                WWt = dot(W, W.T)
                YWt = Yl.dot(W.T)
                # New update for U1
                H1 = Unew + YWt 
                H2 = inv(Y+WWt)
                Unew = dot(H1,H2)
            # Normalize
            if itr == 0:
                lmbda = sqrt((Unew ** 2).sum(axis=0))
            else:
                lmbda = Unew.max(axis=0)
                lmbda[lmbda < 1] = 1
            U[n] = Unew / lmbda


        P = ktensor(U, lmbda)
        
        # Update W
        L1 = inv(dot(U[1].T,U[1]))
        L2 = dot(U[1].T,Yl)
        W = dot(L1,L2)
        BW = dot(U[1],W) 
        normBW = np.linalg.norm(BW)
        ypred = BW
        ypred = np.asarray(ypred)
        if fit_method == 'full':
            normresidual1 = normX ** 2 + P.norm() ** 2 - 2 * P.innerprod(X)
            normresidual2 = normYl ** 2 + normBW ** 2 - 2 * dot(Yl.T,BW)
            normresidual = normresidual1 + normresidual2
            fit = 1 - (normresidual / normX ** 2)
        else:
            fit = itr

        fitchange = abs(fitold - fit)
        print('fitchange:',fitchange)
        exectimes.append(time.clock() - tic)
        #_log.debug(
        #    '[%3d] fit: %.5f | delta: %7.1e | secs: %.5f' %
        #    (itr, fit, fitchange, exectimes[-1])
        #)
        if itr > 0 and fitchange < conv:
            break
    print(ypred)
    ypred[abs(ypred) > 0.5] = 1
    ypred[abs(ypred) < 0.5] = 0 
    print(ypred)
    return P, ypred, fit, itr, array(exectimes)


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


def _init(init, X, N, rank, dtype):
    """
    Initialization for CP models
    """
    Uinit = [None for _ in range(N)]
    if isinstance(init, list):
        Uinit = init
    elif init == 'random':
        for n in range(1, N):
            Uinit[n] = array(rand(X.shape[n], rank), dtype=dtype)
    elif init == 'nvecs':
        for n in range(1, N):
            Uinit[n] = array(nvecs(X, n, rank), dtype=dtype)
    else:
        raise 'Unknown option (init=%s)' % str(init)
    return Uinit

# vim: set et:
