#cython: language_level=3
from scipy.linalg.cython_blas cimport daxpy, ddot
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free


cdef struct DMAT:
    int rows, cols
    double *pointer


cdef DMAT dmat_new(int rows, int cols) nogil:
    cdef DMAT mat
    cdef size_t size = rows * cols * sizeof(double)
    mat.rows = rows
    mat.cols = cols
    mat.pointer = malloc(size)
    if mat.pointer is None:
        raise RuntimeError('could not allocate memory')
    return mat


cdef DMAT dmat_free(DMAT mat) nogil:
    free(mat.pointer)


cdef inline double[:,:] dmat_view(DMAT mat) nogil:
    return <double[:mat.rows,:mat.cols]>(mat.pointer)


cdef void axpy(double a, double[::1] x, double[::1] y) nogil:
    cdef int n = len(x)
    cdef int one = 1
    daxpy(&n, &a, &x[0], &one, &y[0], &one)


cdef double dot(double[::1] x, double[::1] y) nogil:
    cdef int n = len(x)
    cdef int one = 1
    return ddot(&n, &x[0], &one, &y[0], &one)


cdef DMAT subset_xt(double[:,:] X, int[:] xis) nogil:
    cdef int nd = X.shape[1]
    cdef int ni = len(xis)
    cdef DMAT Xt = dmat_new(nd, ni)
    cdef double[:,:] Xtv = <double[:Xt.rows,:Xt.cols]>(Xt.pointer)
    cdef int i, j
    
    for i in range(ni):
        j = xis[i]
        Xtv[:, i] = X[j, :]
    
    return Xt


cdef void rr_solve(double[:,:] X, int[:] xis, double[:] y, double[:] w, double reg, int epochs) nogil:
    """
    RR1 coordinate descent solver.

    Args:
        X: The feature matrix.
        xis: Row numbers in ``X`` that are rated.
        y: Rating values corresponding to ``xis``.
        w: Input/output vector to solve.
    """

    cdef int nd = len(w)
    cdef double[:,:] Xt = subset_xt(X, xis)
    cdef double[::1] resid = y # - matmul(w, Xt)
    cdef double[:] xk
    cdef int e, k
    cdef double num, denom, dw
    
    for e in range(epochs):
        for k in range(nd):
            xk = Xt[k, :]
            num = dot(xk, resid) - reg * w[k]
            denom = dot(xk, xk) + reg
            dw = num / denom
            w[k] += dw
            axpy(-dw, xk, resid)


def train_matrix_cd(mat, this: np.ndarray, other: np.ndarray, reg: float):
    """
    One half of an explicit ALS training round using coordinate descent.

    Args:
        mat: the :math:`m \\times n` matrix of ratings
        this: the :math:`m \\times k` matrix to train
        other: the :math:`n \\times k` matrix of sample features
        reg: the regularization term
    """
    nr = mat.nrows
    nf = other.shape[1]
    assert mat.ncols == other.shape[0]
    assert mat.nrows == this.shape[0]
    assert this.shape[1] == nf

    frob = 0.0

    for i in range(nr):
        cols = mat.row_cs(i)
        if len(cols) == 0:
            continue

        vals = mat.row_vs(i)

        w = this[i, :].copy()
        rr_solve(other, cols, vals, w, reg * len(cols), 2)
        delta = this[i, :] - w
        frob += dot(delta, delta)
        this[i, :] = w

    return np.sqrt(frob)
