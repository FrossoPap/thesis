# nonneg_rescal.py - python script to compute the nonnegative RESCAL tensor
# factorization. Author: Denis Krompass <Denis.Krompass@campus.lmu.de>
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

__version__ = "0.1"
__all__ = ['nonneg_rescal']

__DEF_MAXITER = 500
__DEF_INIT = 'nndsvd'
__DEF_PROJ = True
__DEF_CONV_LS = 5e-4
__DEF_CONV_KL = 1e-4
__DEF_CONV_MUL = 1e-4
__DEF_LMBDA = 0
__DEF_ATTR = []

_log = logging.getLogger('NONNEG_RESCAL')


def nonneg_rescal(X, rank, **kwargs):
    """Non-Negative RESCAL

    Factors a _sparse_ three-way tensor X such that each frontal slice
    X_k = A * R_k * A.T. The frontal slices of a tensor are
    _sparse_ N x N matrices that correspond to the adjecency matrices
    of the relational graph for a particular relation.

    For a full description of the algorithm see:
    [1] Denis Krompass, Maximilian Nickel, Xueyan Jiang, Volker Tresp,
        "Non-Negative Tensor Factorization with RESCAL",
        ECML/PKDD 2013, Prague, Czech Republic

    Parameters
    ----------
    X : list of :class:`scipy.sparse.csr_matrix`
        List of frontal slices X_k of the tensor X. The shape of each X_k is
        n x n.
    rank : int
        Rank of the factorization.
    lambda_A : float, optional
        Regularization parameter for factor matrix A.
        Defaults to 0.
    lambda_R : float, optional
        Regularization parameter for core tensor R.
        Defaults to 0.
    lambda_V : float, optional
        Regularization parameter for the V_l factor matrices of the attributes.
        Defaults to 0.
    attr : list of :class:`scipy.sparse.csr_matrix`, optional
        List of sparse n x v_l attribute matrices. 'v_l' may be different
        for each set of attributes.
        Defaults to None.
    init : string, optional
        Initialization method of the factor matrices. 'nndsvd'
        initializes A based on the NNDSVD algorithm.
        'random' initializes the factor matrices randomly.
        Defaults to 'nndsvd'
    maxIter : int, optional
        Maximium number of iterations of the ALS algorithm.
        Defaults to 500.
    conv : float, optional
        Stop when residual of factorization is less than conv.
        Defaults to 1e-5.
    normalize : boolean, optional
        Keep A normalized during the fit, a L1 regularization penalty is
        employed. *Not implemented for multinomial bases cost function*
        Defautls to `False`-
    costF : string, optional
        Specify the cost function for the fitting:
        'LS' for least squares
        'KL' for generalized Kullback-Leibler Divergence
        'MUL' for multinomial based Kullback-Leibler Divergence
        Defaults to 'LS'.
    verbose : bool, optional
        Show more detailed messages that show the progress of the learning.
        Defaults or `False`.

    Returns
    -------
    A : ndarray
        array of shape ('N', 'rank') corresponding to the factor matrix A
    R : list
        list of 'M' arrays of shape ('rank', 'rank') corresponding to the
        factor matrices R_k
    f : float
        function value of the factorization
    iter : int
        number of iterations until convergence
    exectimes : ndarray
        execution times to compute the updates in each iteration
    """

    # ------------ init options ----------------------------------------------
    ainit = kwargs.pop('init', __DEF_INIT)
    maxIter = kwargs.pop('maxIter', __DEF_MAXITER)
    conv = kwargs.pop('conv', None)
    lmbdaA = kwargs.pop('lambda_A', __DEF_LMBDA)
    lmbdaR = kwargs.pop('lambda_R', __DEF_LMBDA)
    lmbdaV = kwargs.pop('lambda_V', __DEF_LMBDA)
    D = kwargs.pop('attr', __DEF_ATTR)
    dtype = kwargs.pop('dtype', np.float32)
    normalize = kwargs.pop('normalize', False)
    verbose = kwargs.pop('verbose', False)
    costF = kwargs.pop('costF', 'LS')
    if costF == 'LS':
        if conv is None:
            conv = __DEF_CONV_LS
    elif costF == 'KL':
        if conv is None:
            conv = __DEF_CONV_KL
    else:
        if conv is None:
            conv = __DEF_CONV_MUL

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    # ------------- check input -----------------------------------------------
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    for i in xrange(len(X)):
        if not issparse(X[i]):
            raise ValueError('X[%d] is not a sparse matrix' % i)

    sz = X[0].shape
    n = sz[0]
    k = len(X)

    _log.debug('[Config] rank: %d | maxIter: %d | conv: %7.1e |'
               ' lmbda: %7.1e' % (rank, maxIter, conv, lmbdaA))
    _log.debug('[Config] dtype: %s / %s' % (dtype, X[0].dtype))

    # ------- convert X to CSR ------------------------------------------------
    for i in xrange(k):
        X[i] = X[i].tocsr()
        X[i].sort_indices()

    # ---------- initialize A, R and V-----------------------------------------
    _log.debug('Initializing A')
    R = []
    V = []
    if ainit == 'random':
        A = array(rand(n, rank), dtype=dtype)
        for i in xrange(k):
            R.append(array(rand(rank, rank), dtype=dtype))

        for i in xrange(len(D)):
            V.append(array(rand(rank, D[i].shape[1]), dtype=dtype))

    elif ainit == 'nndsvd':
        S = csr_matrix((n, n), dtype=dtype)
        for i in xrange(k):
            S = S + X[i]
            S = S + X[i].T
        A, W = _initialize_nmf(S, rank, 'nndsvd')
        if issparse(A):
            A = A.toarray()
            W = W.toarray()
        else:
            A = np.array(A)
            W = np.array(W)

        Z = np.dot(A.T, W.T)
        for i in xrange(k):
            R.append(Z)

        for i in xrange(len(D)):
            if rank > D[i].shape[1]:
                V.append(array(rand(rank, D[i].shape[1]), dtype=dtype))
            else:
                _, P = _initialize_nmf(D[i], rank, 'nndsvd')
                if issparse(P):
                    P = P.toarray()
                else:
                    P = np.array(P)
                V.append(P)
    else:
        raise 'Unknown init option ("%s")' % ainit

    #  ------ compute factorization -------------------------------------------
    fit = fitchange = fitold = f = 0
    exectimes = []
    if normalize:
        A = __normalize(A)

    for iter in xrange(maxIter):
        tic = time.time()

        fitold = fit
        if costF == 'LS':
            if normalize:
                R = __LS_updateR_L1(X, A, R, lmbdaR)
                A = __LS_updateA_normalized(X, A, R, D, V, lmbdaA)
                V = __LS_updateV_L1(D, A, V, lmbdaV)
            else:
                R = __LS_updateR_L2(X, A, R, lmbdaR)
                A = __LS_updateA(X, A, R, D, V, lmbdaA)
                V = __LS_updateV_L2(D, A, V, lmbdaV)

            # compute fit value
            fit = __LS_compute_fit(X, A, R)

        elif costF == 'KL':
            if normalize:
                R = __KL_updateR(X, A, R, lmbdaR)
                A = __KL_updateA_normalized(X, A, R, D, V, lmbdaA)
                V = __KL_updateV(D, A, V, lmbdaV)
            else:
                R = __KL_updateR(X, A, R, lmbdaR)
                A = __KL_updateA(X, A, R, D, V, lmbdaA)
                V = __KL_updateV(D, A, V, lmbdaV)
            # compute fit value
            fit = __KL_compute_fit(X, A, R)

        elif costF == 'MUL':
            if normalize:
                raise NotImplementedError('Normalize option not implemented '
                                          'for multinomial cost function!')
            else:
                R = __MUL_updateR(X, A, R, lmbdaR)
                A = __MUL_updateA(X, A, R, D, V, lmbdaA)
                V = __MUL_updateV(D, A, V, lmbdaV)
            fit = __MUL_compute_fit(X, A, R)

        fitchange = abs(fitold - fit)
        toc = time.time()
        exectimes.append(toc - tic)

        _log.debug('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f'
                   % (iter, fit, fitchange, exectimes[-1]))
        if iter > 0 and fitchange < conv:
            break

    return A, R, f, iter + 1, array(exectimes)


# ------------------ Updates A ------------------------------------------------
def __LS_updateA(X, A, R, D, V, lmbdaA):
    """Update step for A with LS"""
    eps = np.finfo(np.float).eps
    A[A < eps] = 1.0e-9
    nominator = np.zeros((A.shape), dtype=A.dtype)
    dnominator = np.zeros((A.shape), dtype=A.dtype)
    AtA = np.dot(A.T, A)
    for k in range(len(X)):
        ARt = np.dot(A, R[k].T)
        AR = np.dot(A, R[k])
        nominator = nominator + X[k] * ARt + X[k].T * AR
        dnominator = (dnominator + np.dot(ARt, np.dot(AtA, R[k])) +
                      np.dot(AR, np.dot(AtA, R[k].T)))

    for i in xrange(len(D)):
        nominator += D[i] * V[i].T
        dnominator += dot(A, dot(V[i], V[i].T))

    # update A
    A = A * nominator / (dnominator + lmbdaA * A)
    return A


def __LS_updateA_normalized(X, A, R, D, V, lmbdaA):
    """Update step for normalized A with LS"""
    eps = np.finfo(np.float).eps
    A[A < eps] = 1.0e-9

    nominator = np.zeros((A.shape), dtype=A.dtype)
    dnominator = np.zeros((A.shape), dtype=A.dtype)
    r = R[0].shape[0]
    n = A.shape[0]
    AtA = np.dot(A.T, A)
    for k in range(len(X)):
        ARt = dot(A, R[k].T)
        AR = dot(A, R[k])
        T1 = X[k] * ARt + X[k].T * AR
        T2 = dot(ARt, dot(AtA, R[k])) + dot(AR, dot(AtA, R[k].T))
        nominator = (nominator + T1 +
                     dot(A, np.eye(r, r) * dot(np.ones((1, n)), T2 * A)))
        dnominator = (dnominator + T2 +
                      dot(A, np.eye(r, r) * dot(np.ones((1, n)), T1 * A)))

    for i in xrange(len(D)):
        T1 = D[i] * V[i].T
        T2 = dot(A, dot(V[i], V[i].T))
        nominator += T1 + dot(A, np.eye(r, r) * dot(np.ones((1, n)), T2 * A))
        dnominator += T2 + dot(A, np.eye(r, r) * dot(np.ones((1, n)), T1 * A))

    # update A
    A = __normalize(A * nominator / dnominator)
    return A


def __KL_updateA(X, A, R, D, V, lmbdaA):
    """Update step for A with KL"""
    eps = np.finfo(np.float).eps
    A[A < eps] = 1.0e-9

    E = np.ones((A.shape[0], A.shape[1]))
    nominator = np.zeros((A.shape), dtype=A.dtype)
    dnominator = np.zeros((A.shape), dtype=A.dtype)
    for i in range(len(X)):
        ARi = dot(A, R[i])
        ARit = dot(A, R[i].transpose())
        E_Der = __EMult(ARit + ARi, A.shape[0])
        T1 = __memSavingWHDivision(X[i], dot(A, R[i]), A.T)
        nominator = nominator + (T1 * ARit + T1.transpose() * ARi)
        dnominator = dnominator + E_Der + lmbdaA * E

    for i in xrange(len(D)):
        nominator += __memSavingWHDivision(D[i], A, V[i]) * V[i].T
        dnominator += __EMult(V[i].T, A.shape[0])

    # update A
    A = A * nominator / dnominator
    return A


def __KL_updateA_normalized(X, A, R, D, V, lmbdaA):
    """Update step for normalized A with KL"""
    eps = np.finfo(np.float).eps
    A[A < eps] = 1.0e-9
    nominator = np.zeros((A.shape), dtype=A.dtype)
    dnominator = np.zeros((A.shape), dtype=A.dtype)
    r = R[0].shape[1]
    n = A.shape[0]
    for i in range(len(X)):
        ARi = dot(A, R[i])
        ARit = dot(A, R[i].transpose())
        E_Der = __EMult(ARit + ARi, n)
        S = __memSavingWHDivision(X[i], dot(A, R[i]), A.T)
        S = S * ARit + S.T * ARi
        nominator = (nominator + S +
                     dot(A, np.eye(r, r) * dot(np.ones((1, n)), E_Der * A)))
        dnominator = (dnominator + E_Der +
                      dot(A, np.eye(r, r) * dot(np.ones((1, n)), S * A)))

    for i in xrange(len(D)):
        T1 = __memSavingWHDivision(D[i], A, V[i]) * V[i].T
        T2 = __EMult(V[i].T, n)
        nominator += T1 + dot(A, np.eye(r, r) * dot(np.ones((1, n)), T2 * A))
        dnominator += T2 + dot(A, np.eye(r, r) * dot(np.ones((1, n)), T1 * A))

    # update A
    A = __normalize(A * nominator / dnominator)
    return A


def __MUL_updateA(X, A, R, D, V, lmbdaA):
    """Update step for A with MUL"""
    eps = np.finfo(np.float).eps
    A[A < eps] = 0.0
    n = A.shape[0]
    r = A.shape[1]
    En = np.ones((n, n))
    En1 = np.ones((n, 1))
    nom = np.zeros((n, r))
    dnom = np.zeros((n, r))
    normalizer = np.zeros((n, r))
    for k in xrange(len(R)):
        ARAt = np.dot(A, np.dot(R[k], A.transpose()))
        AR = dot(A, R[k])
        ARt = dot(A, R[k].T)
        T1 = __memSavingWHDivision(X[k], dot(A, R[k]), A.T)
        korpus = T1.dot(ARt) + T1.T.dot(AR)
        no = dot(En1.T, X[k].dot(En1))
        de = dot(dot(En1.T, ARAt), np.ones((n, 1)))
        no[no < eps] = eps
        norm = np.where(de < eps, 0.0, no / de)
        normalizer = dot(En, ARt + AR) * norm
        dnom += normalizer + np.ones((n, r)) * lmbdaA
        nom += korpus

    for i in xrange(len(D)):
        EnD = np.ones((D[i].shape[1], 1))
        AV = np.dot(A, V[i])
        no = dot(dot(En1.T, D[i]), EnD)
        de = dot(dot(En1.T, AV), EnD)
        no[no < eps] = eps
        norm = np.where(de < eps, 0.0, no / de)
        nom += __memSavingWHDivision(D[i], A, V[i]) * V[i].T
        dnom += __EMult(V[i].T, A.shape[0]) * norm

    # update A
    nom[nom < eps] = eps
    mult = np.where(dnom == 0, 0.0, nom / dnom)
    A = A * mult
    A[A < eps] = eps
    return A


# ------------------ Updates R ------------------------------------------------
def __LS_updateR_L1(X, A, R, lmbdaR):
    eps = np.finfo(np.float).eps
    AtA = np.dot(A.T, A)
    for k in range(len(X)):
        R[k][R[k] < eps] = 1.0e-9
        R[k] = (R[k] * (np.dot(A.T * X[k], A) /
                        (np.dot(AtA, np.dot(R[k], AtA)) + lmbdaR *
                         np.ones(R[k].shape, dtype=R[k].dtype))))
    return R


def __LS_updateR_L2(X, A, R, lmbdaR):
    eps = np.finfo(np.float).eps
    AtA = np.dot(A.T, A)
    for k in range(len(X)):
        R[k][R[k] < eps] = 1.0e-9
        R[k] = (R[k] * (np.dot(A.T * X[k], A) /
                        (np.dot(AtA, np.dot(R[k], AtA)) + lmbdaR * R[k])))
    return R


def __KL_updateR(X, A, R, lmbdaR):
    E = np.ones((R[0].shape[0], R[0].shape[1]))  # memory exhaustive
    eps = np.finfo(np.float).eps
    for i in range(len(X)):
        R[i][R[i] < eps] = 1.0e-9
        R[i] = (
            R[i] * dot(A.transpose(),
                       (__memSavingWHDivision(X[i], dot(A, R[i]), A.T) * A)) /
            (dot(A.transpose(), __EMult(A, A.shape[0])) + lmbdaR * E))

    return R


def __MUL_updateR(X, A, R, lmbdaR):
    eps = np.finfo(np.float).eps
    n = A.shape[0]
    E = np.ones((R[0].shape[0], R[0].shape[1]))  # memory exhaustive
    En1 = np.ones((A.shape[0], 1))
    Enn = np.ones((n, n))
    eps = np.finfo(np.float).eps
    for k in xrange(len(R)):
        R[k][R[k] < eps] = 0.0
        ARAt = dot(A, dot(R[k], A.T))
        T1 = __memSavingWHDivision(X[k], dot(A, R[k]), A.T)
        korpus = dot(A.T, T1.dot(A))
        no = En1.T.dot(X[k].dot(En1))
        no[no < eps] = eps
        de = dot(dot(En1.T, ARAt), np.ones((n, 1)))
        norm = np.where(de < eps, 0.0, no / de)
        normalizer = dot(A.T, dot(Enn, A)) * norm
        normalizer += E * lmbdaR
        korpus[korpus == 0] = eps
        mult = np.where(normalizer == 0.0, 0.0, korpus / normalizer)
        R[k] = R[k] * mult
        R[k][R[k] < eps] = eps
    return R


# ------------------ Updates V ------------------------------------------------
def __LS_updateV_L1(D, A, V, lmbdaV):
    if len(D) == 0:
        return V
    _log.debug('Updating V with lambda V: %s' % str(lmbdaV))
    AtA = np.dot(A.T, A)
    for i in range(len(D)):
        V[i] = V[i] * (A.T * D[i] / (np.dot(AtA, V[i]) + lmbdaV *
                                     np.ones(V[i].shape, dtype=V[i].dtype)))
    return V


def __LS_updateV_L2(D, A, V, lmbdaV):
    if len(D) == 0:
        return V

    _log.debug('Updating V with lambda V: %s' % str(lmbdaV))
    AtA = np.dot(A.T, A)
    for i in range(len(D)):
        V[i] = V[i] * (A.T * D[i] / (np.dot(AtA, V[i]) + lmbdaV * V[i]))
    return V


def __KL_updateV(D, A, V, lmbdaV):
    if len(D) == 0:
        return V
    _log.debug('Updating V with KL lambda V: %s' % str(lmbdaV))

    eps = np.finfo(np.float).eps
    for i in range(len(D)):
        E = np.ones(V[0].shape)
        V[i][V[i] < eps] = 1.0e-9
        V[i] = (V[i] * (A.T * __memSavingWHDivision(D[i], A, V[i])) /
                (__EMult(A, V[i].shape[1]).T + lmbdaV * E))
    return V


def __MUL_updateV(D, A, V, lmbdaV):
    if len(D) == 0:
        return V
    eps = np.finfo(np.float).eps
    En1 = np.ones((D[0].shape[0], 1))
    EnD = np.ones((D[0].shape[1], 1))
    for i in range(len(D)):
        V[i][V[i] < eps] = 0.0
        E = np.ones((V[0].shape))
        AV = dot(A, V[i])
        no = dot(dot(En1.T, D[i]), EnD)
        no[no < eps] = eps
        de = dot(dot(En1.T, AV), EnD)
        norm = np.where(de < eps, 0.0, no / de)
        V[i] = (V[i] * (A.T * __memSavingWHDivision(D[i], A, V[i])) /
                (__EMult(A, V[i].shape[1]).T + lmbdaV * E) / norm)
    return V


# ------------------ Compute Fits ---------------------------------------------
def __LS_compute_fit(X, A, R):
    """Compute fit for full slices"""
    f = 0
    # precompute norms of X
    normX = [sum(M.data ** 2) for M in X]
    sumNorm = sum(normX)

    for i in xrange(len(X)):
        ARAt = dot(A, dot(R[i], A.T))
        f += norm(X[i] - ARAt) ** 2
    return 1 - f / sumNorm


def __KL_compute_fit(X, A, R):
    # precompute norms of X
    normX = [sum(M.data) for M in X]
    sumNorm = sum(normX)
    f = 0.0
    for k in xrange(len(X)):
        ARkAt = dot(A, dot(R[k], A.transpose()))
        f += np.sum(X[k] * __log(__memSavingWHDivision(
            X[k], dot(A, R[k]), A.T)).toarray() - X[k] + ARkAt)
    return 1 - f / sumNorm


def __MUL_compute_fit(X, A, R):
    # precompute norms of X
    normX = [sum(M.data) for M in X]
    sumNorm = sum(normX)
    f = 0.0
    for k in xrange(len(X)):
        ARkAt = dot(A, dot(R[k], A.transpose()))
        XdivARAt = __memSavingWHDivision(X[k], dot(A, R[k]), A.T)
        rsum = np.sum(ARkAt)
        logRsum = np.log(rsum)
        score = np.sum((X[k].multiply(__log(XdivARAt)) + X[k] * logRsum).data)
        f += score
    return 1 - f / sumNorm


# ----------------- Helper Functions ------------------------------------------
def __EMult(mat, n):
    # dot(E,mat) where E is nxn and mat nxr
    res = np.zeros((n, mat.shape[1]))
    e = np.ones((1, mat.shape[0]))
    first_row = dot(e, mat)
    for i in xrange(n):
        res[i] = first_row
    return res


def __memSavingWHDivision(X, W, H):
    # solves X / dot(W,H) more memory efficient (X is sparse)
    X = X.tocsr().tocoo()
    row = X.row.copy()
    col = X.col.copy()
    data = X.data.copy()
    r = 0
    entries = []
    div = coo_matrix((data, (row, col)), shape=X.get_shape())
    for i in xrange(len(X.row)):
        if X.row[i] > r:
            if len(entries) > 0:
                Wi = W[r, :]
                for j in xrange(len(entries)):
                    c = X.col[entries[j]]
                    div.data[entries[j]] = dot(Wi, H[:, c])
                    # Divide the X value through the ARiAt value
                    if div.data[entries[j]] > 0.0:
                        div.data[entries[j]] = (
                            X.data[entries[j]] / float(div.data[entries[j]]))
                        if div.data[entries[j]] < 1.0e-9:
                            div.data[entries[j]] = 1.0e-9
                    else:
                        div.data[entries[j]] = 1.0e-9
            r = X.row[i]
            entries = [i]
        elif X.row[i] == r:
            entries.append(i)
        else:
            raise(Exception("Smat is not sorted by rows!"))
    else:
        if len(entries) > 0:
            Wi = W[r, :]
            for j in xrange(len(entries)):
                c = X.col[entries[j]]
                div.data[entries[j]] = dot(Wi, H[:, c])
                # Divide the X value through the ARiAt value
                if div.data[entries[j]] > 0.0:
                    div.data[entries[j]] = (
                        X.data[entries[j]] / float(div.data[entries[j]]))
                    if div.data[entries[j]] < 1.0e-9:
                        div.data[entries[j]] = 1.0e-9
                else:
                    div.data[entries[j]] = 1.0e-9
    return div.tocsr()


def __normalize(M):
    # normalize column vectors of M to a length of 1
    for j in xrange(M.shape[1]):
        M[:, j] = M[:, j] / np.linalg.norm(M[:, j])
    return M


def __log(X):
    X = X.tocoo()
    X.data = np.log(X.data)
    return X.tocsr()


if __name__ == '__main__':
    """Simple Test if the the code is running"""
    X = [csr_matrix(np.random.randint(0, 2, size=(10, 10)).astype(np.float32))
         for i in xrange(6)]
    nonneg_rescal(X, 3, verbose=True, normalize=True)
    nonneg_rescal(X, 3, verbose=True)
    nonneg_rescal(X, 3, verbose=True, normalize=True, costF='KL')
    nonneg_rescal(X, 3, verbose=True, costF='KL')
    nonneg_rescal(X, 3, verbose=True, costF='MUL')
    nonneg_rescal(X, 3, verbose=True, lambda_A=0.01, lambda_R=0.01)
