# cython: language_level=3str

from cython.parallel cimport prange
from libc.math cimport sqrt

import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_blas cimport daxpy, ddot
from scipy.linalg.cython_lapack cimport dposv

from lenskit.util.csmatrix cimport CSMatrix


cdef void easy_axpy(double a, double[::1] x, double[::1] y) noexcept nogil:
    cdef int n = x.shape[0]
    cdef int one = 1
    daxpy(&n, &a, &x[0], &one, &y[0], &one)


cdef int easy_dposv(double[:,::1] A, double[::1] b) except -1 nogil:
    cdef char uplo = 'U'
    cdef int n = b.shape[0]
    cdef int nrhs = 1
    cdef int info = 0
    dposv(&uplo, &n, &nrhs, &A[0,0], &n, &b[0], &n, &info)
    if info < 0:
        raise ValueError('invalid argument')
    if info > 0:
        raise RuntimeError('dposv failed')

    return 0


cdef double easy_dot(double[::1] x, double[::1] y) noexcept nogil:
    cdef int n = len(x)
    cdef int one = 1
    return ddot(&n, &x[0], &one, &y[0], &one)


cdef int _rr_solve(double[:,::1] X, int[::1] xis, float[::1] y, double[::1] w, double reg, int epochs) except -1 nogil:
    """
    RR1 coordinate descent solver.

    Args:
        X(ndarray): The feature matrix.
        xis(ndarray): Row numbers in ``X`` that are rated.
        y(ndarray): Rating values corresponding to ``xis``.
        w(ndarray): Input/output vector to solve.
    """

    cdef int nd = len(w)
    cdef int e, k
    cdef double num, denom, dw
    cdef double[::1,:] Xt = X.T
    cdef double[::1] xk

    cdef double[:,::1] Xss = np.take(Xt, xis, axis=1)
    cdef double[::1] resid = w @ Xss
    resid *= -1.0
    resid += y

    for e in range(epochs):
        for k in range(nd):
            xk = Xss[k, :]
            num = easy_dot(xk, resid) - reg * w[k]
            denom = easy_dot(xk, xk) + reg
            dw = num / denom
            w[k] += dw
            easy_axpy(-dw, xk, resid)

    return 0


cpdef double _train_matrix_cd(CSMatrix mat, double[:,::1] this, double[:,::1] other, double reg) except? -1.0 nogil:
    """
    One half of an explicit ALS training round using coordinate descent.

    Args:
        mat: the :math:`m \\times n` matrix of ratings
        this: the :math:`m \\times k` matrix to train
        other: the :math:`n \\times k` matrix of sample features
        reg: the regularization term
    """
    cdef int nr = mat.nrows
    cdef int nf = other.shape[1]
    cdef int sp, ep, n, i
    cdef double[::1] w, vals, double

    assert mat.ncols == other.shape[0]
    assert mat.nrows == this.shape[0]
    assert this.shape[1] == nf

    cdef double frob = 0.0

    for i in range(nr):
        sp, ep = mat.row_extent(i)
        n = ep - sp
        if n == 0:
            continue

        vals = mat.values[sp:ep]

        w = this[i, :].copy()
        _rr_solve(other, cols, vals, w, reg * n, 2)
        delta = this[i, :] - w
        frob += easy_dot(delta, delta)
        this[i, :] = w

    return sqrt(frob)


# cpdef double train_matrix_lu(CSMatrix mat, double[:,::1] this, double[:,::1] other, double reg) nogil:
#     """
#     One half of an explicit ALS training round using LU-decomposition on the normal
#     matrices to solve the least squares problem.

#     Args:
#         mat: the :math:`m \\times n` matrix of ratings
#         this: the :math:`m \\times k` matrix to train
#         other: the :math:`n \\times k` matrix of sample features
#         reg: the regularization term
#     """
#     cdef int nr = mat.nrows
#     cdef int nf = other.shape[1]
#     cdef double[:,::1] regI = np.identity(nf) * reg
#     cdef int sp, ep, n
#     cdef int[::1] cols
#     cdef double[::1] vals
#     cdef double[:,:] M, MMT, A
#     cdef double[::1] V
#     cdef double frob = 0.0

#     assert mat.ncols == other.shape[0]

#     for i in prange(nr):
#         sp, ep = mat.row_extent(i)
#         n = ep - sp
#         if n == 0:
#             continue

#         cols = mat.colind[sp:ep]
#         vals = mat.values[sp:ep]

#         M = other[cols, :]
#         MMT = M.T @ M
#         # assert MMT.shape[0] == ctx.n_features
#         # assert MMT.shape[1] == ctx.n_features
#         A = MMT + regI * len(cols)
#         V = M.T @ vals
#         # and solve
#         _dposv(A, V, True)
#         delta = this[i, :] - V
#         frob += np.dot(delta, delta)
#         this[i, :] = V

#     return np.sqrt(frob)


# cpdef double[::1] train_bias_row_lu(int[::1] items, float[::1] ratings, double[:,::1] other, double reg) nogil:
#     """
#     Args:
#         items(np.ndarray[i64]): the item IDs the user has rated
#         ratings(np.ndarray): the user's (normalized) ratings for those items
#         other(np.ndarray): the item-feature matrix
#         reg(float): the regularization term
#     Returns:
#         np.ndarray: the user-feature vector (equivalent to V in the current LU code)
#     """
#     cdef double[:,:] M = other[items, :]
#     cdef int nf = other.shape[1]
#     cdef double[:,::1] regI = np.identity(nf) * reg
#     cdef double[:,::1] MMT = M.T @ M
#     cdef double[:,::1] A = MMT + regI * len(items)

#     cdef double[:,::1] V = M.T @ ratings
#     easy_dposv(A, V)

#     return V

