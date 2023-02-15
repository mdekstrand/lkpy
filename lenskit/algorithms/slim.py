"""
The SLIM algorithm.
"""

import logging
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

from csr import CSR
from lenskit.data.matrix import sparse_ratings
from . import Predictor

try:
    from sklearn.linear_model._cd_fast import sparse_enet_coordinate_descent
    SKL_AVAILABLE = True
except ImportError:
    SKL_AVAILABLE = False

_logger = logging.getLogger(__name__)

class SLIMTrainContext(NamedTuple):
    n_items: int
    n_users: int
    matrix: CSR
    sparse: bool
    l1: float
    l2: float
    rng: np.random.RandomState


def _train_item(ctx: SLIMTrainContext, item: int):
    if ctx.sparse:
        # get the ratings for this item
        t = ctx.matrix.transpose()
        i_users = t.row_cs(item)
        y = t.row_vs(item)
        del t

        # get the users' other ratings
        X = ctx.matrix.to_scipy()
        X = X[i_users, :]

        # zero out training for this item
        X.data[X.indices == item] = 0

    else:
        # convert and extract training matrix
        X = ctx.matrix.copy(copy_structure=False).to_scipy()
        y = X[:, item].toarray()

        # zero out training ratings for this item
        X.data[X.indices == item] = 0

    nu = len(y)
    w = np.zeros(ctx.n_items)
    Xmean = np.zeros(ctx.n_items)
    w, gap, tol, n_iter = sparse_enet_coordinate_descent(w, ctx.l1 / nu, ctx.l2 / nu,
                                                         X.data, X.indices, X.indptr, y,
                                                         None, Xmean, 1000, 1.0e-4,
                                                         ctx.rng, 1, 1)

class SLIM(Predictor):
    """
    SLIM (sparse linear methods) collaborative filtering with either explicit or implicit
    feedback :cite:p:`Ning2011-qu`.

    Args:
        l1:
            the :math:`L_1` regularization strength.
        l2:
            the :math:`L_2` regularization strength.

        feedback(str):
            Control how feedback should be interpreted.  Specifies defaults for the other
            settings, which can be overridden individually; can be one of the following values:

            ``explicit``
                Configure for explicit-feedback mode: use rating values.  This is the default
                for compatibility with :class:`~lenskit.algorithms.item_knn.ItemItem`.

            ``implicit``
                Configure for implicit-feedback mode: ignore rating values and predict 1/0 instead.

        sparse_training(bool):
            whether to treat the training data as sparse (``True``) or dense (``False``, the default).
            Don't use sparse with implicit feedback data.

        center(bool):
            whether to normalize (mean-center) rating vectors prior to learning and predicting
            user-item scores.  Defaults to ``False``.  When combined with sparse data, the
            missing observations are still treated as 0; this amounts to assuming that they
            are average ratings.  Don't use this with implicit-feedback data.

        min_ratings(int):
            items with fewer than ``min_ratings` ratings are excluded from training.  This
            prunes the item space and decreases coverage as well as training time.

    Attributes:
        item_index_(pandas.Index): the index of item IDs.
        item_means_(numpy.ndarray): the mean rating for each known item (if necessary)
        weights_(matrix.CSR): the learned item-item weight matrix
        user_index_(pandas.Index): the index of known user IDs.
        rating_matrix_(matrix.CSR): the user-item rating matrix for looking up users' ratings.
    """

    def __init__(self, *, l1=1.0, l2=1.0, feedback=None, sparse_training=False, center=False,
                 min_ratings=1):
        self.l1 = l1
        self.l2 = l2
        if feedback is None:
            _logger.warn('no feedback mode specified, assuming explicit')
            warnings.warn('no feedback mode specified, assuming explicit')
        elif feedback not in ('implicit', 'explicit'):
            raise ValueError(f'invalid feedback type {feedback}')
        else:
            self.feedback = feedback

        self.sparse_training = sparse_training
        self.center = center
        self.min_ratings = min_ratings

    def fit(self, ratings, **kwargs):
        if self.feedback == 'implicit' and 'rating' in ratings.columns:
            ratings = ratings[['user', 'item']]

        rmat, users, items = sparse_ratings(ratings)
        _logger.info('training SLIM for %d users, %d items, %d ratings',
                     len(users), len(items), rmat.nnz)
        self.rating_matrix_ = rmat
        self.item_index_ = items
        self.user_index_ = users

        if self.center:
            rmat = rmat.transpose()
            self.item_means_ = rmat.normalize_rows('center')
            rmat = rmat.transpose()

    def predict_for_user(self, user, items, ratings=None):
        if ratings is not None:
            rvec = ratings.reindex(self.items, fill_value=0).values
        else:
            if user not in self.user_index_:
                _logger.debug('user %s missing, returning empty predictions', user)
                return pd.Series(np.nan, index=items)
            upos = self.user_index_.get_loc(user)
            # get the full rating vector
            rvec = self.rating_matric_.row(upos)

        # compute all predictions - is this too slow?
        preds = self.weights_.multiply(rvec)
        preds = pd.Series(preds, index=self.items)
        preds = preds.reindex(items)

        return preds
