import numpy as np


def _inplace_axpy(a, x, y):
    for i in range(len(x)):
        y[i] += a * x[i]


def _cg_a_mult(OtOr, X, y, v):
    """
    Compute the multiplication Av, where A = X'X + X'yX + λ.
    """
    XtXv = OtOr @ v
    XtyXv = X.T @ (y * (X @ v))
    return XtXv + XtyXv


def _cg_solve(OtOr, X, y, w, epochs):
    """
    Use conjugate gradient method to solve the system M†(X'X + X'yX + λ)w = M†X'(y+1).
    The parameter OtOr = X'X + λ.
    """
    nf = X.shape[1]
    # compute inverse of the Jacobi preconditioner
    Ad = np.diag(OtOr).copy()
    for i in range(X.shape[0]):
        for k in range(nf):
            Ad[k] += X[i, k] * y[i] * X[i, k]

    iM = np.reciprocal(Ad)

    # compute residuals
    b = X.T @ (y + 1.0)
    r = _cg_a_mult(OtOr, X, y, w)
    r *= -1
    r += b

    # compute initial values
    z = iM * r
    p = z

    # and solve
    for i in range(epochs):
        gam = np.dot(r, z)
        Ap = _cg_a_mult(OtOr, X, y, p)
        al = gam / np.dot(p, Ap)
        _inplace_axpy(al, p, w)
        _inplace_axpy(-al, Ap, r)
        z = iM * r
        bet = np.dot(r, z) / gam
        p = z + bet * p


def _implicit_otor(other, reg):
    nf = other.shape[1]
    regmat = np.identity(nf)
    regmat *= reg
    Ot = other.T
    OtO = Ot @ other
    OtO += regmat
    return OtO


def train_implicit_cg(rps, cis, vs, this: np.ndarray, other: np.ndarray, reg: float):
    "One half of an implicit ALS training round with conjugate gradient."
    nr = len(rps) - 1
    nc = other.shape[0]

    OtOr = _implicit_otor(other, reg)

    frob = 0.0

    for i in range(nr):
        rs = rps[i]
        re = rps[i + 1]
        cols = cis[rs:re]
        rates = vs[rs:re]
        if len(cols) == 0:
            continue

        # we can optimize by only considering the nonzero entries of Cu-I
        # this means we only need the corresponding matrix columns
        M = other[cols, :]
        # and solve
        w = this[i, :].copy()
        _cg_solve(OtOr, M, rates, w, 3)

        # update stats
        delta = this[i, :] - w
        frob += np.dot(delta, delta)

        # put back the result
        this[i, :] = w

    return np.sqrt(frob)


# mat, this: np.ndarray, other: np.ndarray, reg: float
# pythran export train_implicit_cg(int64[:], float64[:], float64[:], float64[:,:], float64[:,:], float64)
