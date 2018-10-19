"""
Poisson matrix factorization.
"""

from collections import namedtuple
import logging

import numpy as np
import pandas as pd
from scipy import special

from .. import util, matrix
from . import Predictor, Trainable
from .mf_common import MFModel

_logger = logging.getLogger(__name__)

PFHP = namedtuple('PFHP', [
    'v_shape',  # a  c
    'a_shape',  # a' c'
    'a_mean'    # b' d'
])
PFHP.___doc__ = """
Poisson factorization hyper-parameter set.

Attributes:
    v_shape(double):
        the prior for the value shape parameter (:math:`a` or :math:`c`)
    a_shape(double):
        the prior for the activity shape parameter (:math:`a'` or :math:`c'`)
    a_mean(double):
        the prior for the activity mean parameter (:math:`b'` or :math:`d'`)
"""

_PFData = namedtuple('_PFData', [
    'nusers', 'nitems', 'nnz',
    'users', 'items', 'ratings',
    'ui_csr', 'iu_csr'
])
_PFData.__doc__ = """
Internal data set for training Poisson factorization models.

Attributes:
    nusers(int): the number of users
    nitems(int): the number of items
    nnz(int): the numer of ratings
    users(numpy.ndarray): rating user IDs
    items(numpy.ndarray): rating item IDs
    ratings(numpy.ndarray): rating values
    ui_csr(lenskit.matrix.CSR):
        User-item CSR matrix, with value pointers indexing into ratings.
    iu_csr(lenskit.matrix.CSR):
        Item-user CSR matrix, with value pointers indexing into ratings.
"""

_PFHalfModel = namedtuple('_PFHalfModel', [
    'val_shape', 'val_rate',
    'act_shape', 'act_rate'
])
_PFHalfModel.__doc__ = """
Half of an internal model for Poisson factorization, representing user or item
shape and rate parameters.

Attributes:
    val_shape(numpy.ndarray): value shape matrix (:math:`n \\times K`)
    val_rate(numpy.ndarray): value shape matrix (:math:`n \\times K`)
    act_shape(numpy.ndarray): activity shape vector (:math:`n`)
    act_rate(numpy.ndarray): activity shape vector (:math:`n`)
"""


class HPF(Predictor, Trainable):
    """
    Hierarchical Poisson factorization [GHB2013]_.

    .. [GHB2013] Prem Gopalan, Jake M. Hofman, and David M. Blei. 2013.
        Scalable Recommendation with Poisson Factorization.
        arXiv:1311.1704 [cs, stat] (November 2013). Retrieved February 9, 2017
        from http://arxiv.org/abs/1311.1704.

    Args:
        features(int): the number of features to train
        iterations(int): the number of iterations to train
        uhp(PFHP): user hyperparameters
        ihp(PFHP): item hyperparameters
    """
    timer = None

    def __init__(self, features, iterations=20, uhp=PFHP(0.3, 0.3, 1), ihp=PFHP(0.3, 0.3, 1)):
        self.features = features
        self.iterations = iterations
        self.user_hp = uhp
        self.item_hp = ihp

    def train(self, ratings, bias=None):
        """
        Train an HPF model.

        Args:
            ratings: the ratings data frame.

        Returns:
            MFModel: The trained F model.
        """
        self.timer = util.Stopwatch()

        _logger.info('[%s] initializing Poisson training', self.timer)

        users = pd.Index(ratings.user.unique())
        nusers = len(users)
        items = pd.Index(ratings.item.unique())
        nitems = len(items)

        u_inds = users.get_indexer(ratings.user)
        i_inds = items.get_indexer(ratings.item)
        rates = np.require(ratings.rating.values, np.float64)

        ui_csr = matrix.csr_from_coo(u_inds, i_inds, shape=(nusers, nitems), pointers=True)
        iu_csr = matrix.csr_from_coo(i_inds, u_inds, shape=(nitems, nusers), pointers=True)

        data = _PFData(nusers, nitems, len(rates), u_inds, i_inds, rates, ui_csr, iu_csr)

        u_mod = self._init_params(nusers, self.user_hp)
        i_mod = self._init_params(nitems, self.item_hp)

        _logger.info('[%s] training Poisson MF model for %d features',
                     self.timer, self.features)
        for epoch in range(self.iterations):
            u_mod, i_mod = self._train_iter(data, u_mod, i_mod)

        _logger.info('[%s] finalizing Poisson MF model')
        umat = u_mod.val_shape / u_mod.val_rate
        imat = i_mod.val_shape / i_mod.val_rate

        _logger.info('trained model in %s', self.timer)

        return MFModel(users, items, umat, imat)

    def _init_params(self, n, hp):
        v_shp = np.random.randn(n, self.features) * 0.01
        v_shp += hp.v_shape
        v_rte = np.random.randn(n, self.features) * 0.01
        v_rte += hp.a_shape  # FIXME double-check this
        a_shp = np.random.randn(n) * 0.01 + hp.a_shape
        a_rte = np.full(n, hp.a_shape + self.features * hp.v_shape)

        return _PFHalfModel(v_shp, v_rte, a_shp, a_rte)

    def _train_iter(self, data: _PFData, umod: _PFHalfModel, imod: _PFHalfModel):
        phi = self._make_variational_matrix(data, umod, imod)

        umod = self._train_model(data, data.ui_csr, phi, umod, imod, self.user_hp)
        imod = self._train_model(data, data.iu_csr, phi, imod, umod, self.item_hp)

        return umod, imod

    def _make_variational_matrix(self, data, umod, imod):
        _logger.debug('[%s] making variational matrix', self.timer)
        uis = data.users
        iis = data.items

        phi = np.empty((data.nnz, self.features))
        tmp = np.empty((data.nnz, self.features))

        # compute user shape contributions
        special.digamma(umod.val_shape[uis, :], out=phi)
        np.log(umod.val_rate[uis, :], out=tmp)
        phi -= tmp

        # compute item shape contributions
        special.digamma(imod.val_shape[iis, :], out=tmp)
        phi += tmp
        np.log(imod.val_rate[iis, :], out=tmp)
        phi -= tmp

        # exponentiate, and we're done with tmp
        del tmp
        np.exp(phi, out=phi)

        # and normalize to sum to 1
        rsums = np.sum(phi, axis=1)
        phi /= rsums.reshape((data.nnz, 1))

        return phi

    def _train_model(self, data, csr, phi, prev, other, hp):
        nrows = prev.val_shape.shape[0]
        _logger.debug('[%s] computing model update for %d rows', self.timer, nrows)
        v_shp = np.empty(prev.val_shape.shape)
        v_rte = np.empty(prev.val_rate.shape)
        a_rte = np.empty(prev.act_rate.shape)

        for r in range(nrows):
            # extract location data to look up results
            cols = csr.row_cs(r)
            ncs = len(cols)
            vps = csr.row_vps(r)
            ys = data.ratings[vps]
            phis = phi[vps, :]

            # compute value shape update
            vs_vec = np.sum(ys.reshape((ncs, 1)) * phis, axis=0)
            vs_vec += hp.v_shape
            v_shp[r, :] = vs_vec

            # compute value rate update
            vr_vec = np.sum(other.val_shape[cols, :] / other.val_rate[cols, :], axis=0)
            vr_vec += prev.act_shape[r] / prev.act_rate[r]
            v_rte[r, :] = vr_vec

            # compute activity rate update
            a_rte[r] = hp.a_shape / hp.a_mean + np.sum(vs_vec / vr_vec)

        return _PFHalfModel(v_shp, v_rte, prev.act_shape, a_rte)

    def predict(self, model: MFModel, user, items, ratings=None):
        # look up user index
        return model.score_by_ids(user, items)
