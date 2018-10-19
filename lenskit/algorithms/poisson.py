"""
Poisson matrix factorization.
"""

from collections import namedtuple
import logging

import numpy as np

from .. import util, matrix
from . import Predictor, Trainable
from .mf_common import MFModel

_logger = logging.getLogger(__name__)

PFHP = namedtuple('PFHP', [
    'v_shape', # a/c
    'a_shape', # a'/c'
    'a_mean'   # b'/d'
])


class HPF(Predictor, Trainable):
    """
    Hierarchical Poisson factorization.

    .. [ZWSP2008] Yunhong Zhou, Dennis Wilkinson, Robert Schreiber, and Rong Pan. 2008.
        Large-Scale Parallel Collaborative Filtering for the Netflix Prize.
        In +Algorithmic Aspects in Information and Management_, LNCS 5034, 337–348.
        DOI `10.1007/978-3-540-68880-8_32 <http://dx.doi.org/10.1007/978-3-540-68880-8_32>`_.

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

        _logger.info('[%s] initializing Poisson training', self.timer, self.features)
        rmat = matrix.sparse_ratings(ratings)

        uv_shp, uv_rte, ua_shp, ua_rte = self._init_params(self.user_hp)
        iv_shp, iv_rte, ip_shp, ip_rte = self._init_params(self.item_hp)

        _logger.info('[%s] training Poisson MF model for %d features',
                     self.timer, self.features)
        for epoch, model in enumerate(self._train_iters(current, uctx, ictx)):
            current = model

        _logger.info('trained model in %s', self.timer)

        return current

    def _init_params(self, n, hp):
        v_shp = np.random.randn(n, self.features) * 0.01
        v_shp += hp.w_shape
        v_rte = np.random.randn(n, self.features) * 0.01
        v_rte += hp.a_shape  # double-check
        a_shp = np.random.randn(n) * 0.01 + hp.a_shape
        a_rte = np.full(hp.a_shape + self.features * hp.v_shape, n)

        return v_shp, v_rte, a_shp, a_rte
