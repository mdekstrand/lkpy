import logging
from typing import Literal

import numpy as np
from csr import CSR
from numba import njit, prange
from seedbank import numpy_rng

from ... import util
from ...data import sparse_ratings
from ...math.solve import _dposv
from ..bias import Bias
from ..mf_common import MFPredictor
from .common import PartialModel, inplace_axpy

_logger = logging.getLogger(__name__)


@njit
def _rr_solve(X, xis, y, w, reg, epochs):
    """
    RR1 coordinate descent solver.

    Args:
        X(ndarray): The feature matrix.
        xis(ndarray): Row numbers in ``X`` that are rated.
        y(ndarray): Rating values corresponding to ``xis``.
        w(ndarray): Input/output vector to solve.
    """

    nd = len(w)
    Xt = X.T[:, xis]
    resid = w @ Xt
    resid *= -1.0
    resid += y

    for e in range(epochs):
        for k in range(nd):
            xk = Xt[k, :]
            num = np.dot(xk, resid) - reg * w[k]
            denom = np.dot(xk, xk) + reg
            dw = num / denom
            w[k] += dw
            inplace_axpy(-dw, xk, resid)


@njit(parallel=True, nogil=True)
def _train_matrix_cd(mat: CSR, this: np.ndarray, other: np.ndarray, reg: float):
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

    for i in prange(nr):
        cols = mat.row_cs(i)
        if len(cols) == 0:
            continue

        vals = mat.row_vs(i)

        w = this[i, :].copy()
        _rr_solve(other, cols, vals, w, reg * len(cols), 2)
        delta = this[i, :] - w
        frob += np.dot(delta, delta)
        this[i, :] = w

    return np.sqrt(frob)


@njit(parallel=True, nogil=True)
def _train_matrix_lu(mat, this: np.ndarray, other: np.ndarray, reg: float):
    """
    One half of an explicit ALS training round using LU-decomposition on the normal
    matrices to solve the least squares problem.

    Args:
        mat: the :math:`m \\times n` matrix of ratings
        this: the :math:`m \\times k` matrix to train
        other: the :math:`n \\times k` matrix of sample features
        reg: the regularization term
    """
    nr = mat.nrows
    nf = other.shape[1]
    regI = np.identity(nf) * reg
    assert mat.ncols == other.shape[0]
    frob = 0.0

    for i in prange(nr):
        cols = mat.row_cs(i)
        if len(cols) == 0:
            continue

        vals = mat.row_vs(i)
        M = other[cols, :]
        MMT = M.T @ M
        # assert MMT.shape[0] == ctx.n_features
        # assert MMT.shape[1] == ctx.n_features
        A = MMT + regI * len(cols)
        V = M.T @ vals
        # and solve
        _dposv(A, V, True)
        delta = this[i, :] - V
        frob += np.dot(delta, delta)
        this[i, :] = V

    return np.sqrt(frob)


@njit(nogil=True)
def _train_bias_row_lu(items, ratings, other, reg):
    """
    Args:
        items(np.ndarray[i64]): the item IDs the user has rated
        ratings(np.ndarray): the user's (normalized) ratings for those items
        other(np.ndarray): the item-feature matrix
        reg(float): the regularization term
    Returns:
        np.ndarray: the user-feature vector (equivalent to V in the current LU code)
    """
    M = other[items, :]
    nf = other.shape[1]
    regI = np.identity(nf) * reg
    MMT = M.T @ M
    A = MMT + regI * len(items)

    V = M.T @ ratings
    _dposv(A, V, True)

    return V


class BiasedMF(MFPredictor):
    """
    Biased matrix factorization trained with alternating least squares :cite:p:`Zhou2008-bj`.  This
    is a prediction-oriented algorithm suitable for explicit feedback data, using the alternating
    least squares approach to compute :math:`P` and :math:`Q` to minimize the regularized squared
    reconstruction error of the ratings matrix.

    It provides two solvers for the optimization step (the `method` parameter):

    ``'cd'`` (the default)
        Coordinate descent :cite:p:`Takacs2011-ix`, adapted for a separately-trained bias model and
        to use weighted regularization as in the original ALS paper :cite:p:`Zhou2008-bj`.
    ``'lu'``
        A direct implementation of the original ALS :cite:p:`Zhou2008-bj` using LU-decomposition
        to solve for the optimized matrices.

    See the base class :class:`.MFPredictor` for documentation on
    the estimated parameters you can extract from a trained model.

    Args:
        features(int): the number of features to train
        iterations(int): the number of iterations to train
        reg(float): the regularization factor; can also be a tuple ``(ureg, ireg)`` to
            specify separate user and item regularization terms.
        damping(float): damping factor for the underlying bias.
        bias(bool or :class:`Bias`): the bias model.  If ``True``, fits a :class:`Bias` with
            damping ``damping``.
        method(str): the solver to use (see above).
        rng_spec:
            Random number generator or state (see :func:`seedbank.numpy_rng`).
        progress: a :func:`tqdm.tqdm`-compatible progress bar function
    """

    timer = None
    bias: Bias | None | Literal[False]

    def __init__(
        self,
        features,
        *,
        iterations=20,
        reg=0.1,
        damping=5,
        bias=True,
        method="cd",
        rng_spec=None,
        progress=None,
        save_user_features=True,
    ):
        self.features = features
        self.iterations = iterations
        self.regularization = reg
        self.damping = damping
        self.method = method
        if bias is True:
            self.bias = Bias(damping=damping)
        else:
            self.bias = bias
        self.progress = progress if progress is not None else util.no_progress
        self.rng = numpy_rng(rng_spec)
        self.save_user_features = save_user_features

    def fit(self, ratings, **kwargs):
        """
        Run ALS to train a model.

        Args:
            ratings: the ratings data frame.

        Returns:
            The algorithm (for chaining).
        """
        util.check_env()
        self.timer = util.Stopwatch()

        for epoch, algo in enumerate(self.fit_iters(ratings, **kwargs)):
            pass  # we just need to do the iterations

        if self.user_features_ is not None:
            _logger.info(
                "trained model in %s (|P|=%f, |Q|=%f)",
                self.timer,
                np.linalg.norm(self.user_features_, "fro"),
                np.linalg.norm(self.item_features_, "fro"),
            )
        else:
            _logger.info(
                "trained model in %s (|Q|=%f)",
                self.timer,
                np.linalg.norm(self.item_features_, "fro"),
            )

        del self.timer
        return self

    def fit_iters(self, ratings, **kwargs):
        """
        Run ALS to train a model, returning each iteration as a generator.

        Args:
            ratings: the ratings data frame.

        Returns:
            The algorithm (for chaining).
        """

        if self.bias:
            _logger.info("[%s] fitting bias model", self.timer)
            self.bias.fit(ratings)

        current, uctx, ictx = self._initial_model(ratings)

        _logger.info(
            "[%s] training biased MF model with ALS for %d features", self.timer, self.features
        )
        for epoch, model in enumerate(self._train_iters(current, uctx, ictx)):
            self._save_params(model)
            yield self

    def _save_params(self, model):
        "Save the parameters into model attributes."
        self.item_index_ = model.items
        self.user_index_ = model.users
        self.item_features_ = model.item_matrix
        if self.save_user_features:
            self.user_features_ = model.user_matrix
        else:
            self.user_features_ = None

    def _initial_model(self, ratings):
        # transform ratings using offsets
        if self.bias:
            _logger.info("[%s] normalizing ratings", self.timer)
            ratings = self.bias.transform(ratings)

        "Initialize a model and build contexts."
        rmat, users, items = sparse_ratings(ratings)
        n_users = len(users)
        n_items = len(items)

        _logger.debug("setting up contexts")
        trmat = rmat.transpose()

        _logger.debug("initializing item matrix")
        imat = self.rng.standard_normal((n_items, self.features))
        imat /= np.linalg.norm(imat, axis=1).reshape((n_items, 1))
        _logger.debug("|Q|: %f", np.linalg.norm(imat, "fro"))
        _logger.debug("initializing user matrix")
        umat = self.rng.standard_normal((n_users, self.features))
        umat /= np.linalg.norm(umat, axis=1).reshape((n_users, 1))
        _logger.debug("|P|: %f", np.linalg.norm(umat, "fro"))

        return PartialModel(users, items, umat, imat), rmat, trmat

    def _train_iters(self, current, uctx, ictx):
        """
        Generator of training iterations.

        Args:
            current(PartialModel): the current model step.
            uctx(ndarray): the user-item rating matrix for training user features.
            ictx(ndarray): the item-user rating matrix for training item features.
        """
        n_items = len(current.items)
        n_users = len(current.users)
        assert uctx.nrows == n_users
        assert uctx.ncols == n_items
        assert ictx.nrows == n_items
        assert ictx.ncols == n_users

        if self.method == "cd":
            train = _train_matrix_cd
        elif self.method == "lu":
            train = _train_matrix_lu
        else:
            raise ValueError("invalid training method " + self.method)

        if isinstance(self.regularization, tuple):
            ureg, ireg = self.regularization
        else:
            ureg = ireg = self.regularization

        for epoch in self.progress(range(self.iterations), desc="BiasedMF", leave=False):
            du = train(uctx, current.user_matrix, current.item_matrix, ureg)
            _logger.debug("[%s] finished user epoch %d", self.timer, epoch)
            di = train(ictx, current.item_matrix, current.user_matrix, ireg)
            _logger.debug("[%s] finished item epoch %d", self.timer, epoch)
            _logger.info("[%s] finished epoch %d (|ΔP|=%.3f, |ΔQ|=%.3f)", self.timer, epoch, du, di)
            yield current

    def predict_for_user(self, user, items, ratings=None):
        scores = None
        u_offset = None
        if ratings is not None and len(ratings) > 0:
            if self.bias:
                ratings, u_offset = self.bias.transform_user(ratings)

            ri_idxes = self.item_index_.get_indexer_for(ratings.index)
            ri_good = ri_idxes >= 0
            ri_it = ri_idxes[ri_good]
            ri_val = ratings.values[ri_good]

            # unpack regularization
            if isinstance(self.regularization, tuple):
                ureg, ireg = self.regularization
            else:
                ureg = self.regularization

            u_feat = _train_bias_row_lu(ri_it, ri_val, self.item_features_, ureg)
            scores = self.score_by_ids(user, items, u_feat)
        else:
            # look up user index
            scores = self.score_by_ids(user, items)

        if self.bias and ratings is not None and len(ratings) > 0:
            return self.bias.inverse_transform_user(user, scores, u_offset)
        elif self.bias:
            return self.bias.inverse_transform_user(user, scores)
        else:
            return scores

    def __str__(self):
        return "als.BiasedMF(features={}, regularization={})".format(
            self.features, self.regularization
        )
