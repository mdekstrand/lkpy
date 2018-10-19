import os
import logging
from pathlib import Path

from lenskit.algorithms import poisson

import pandas as pd
import numpy as np

import pytest
from pytest import approx, mark

import lk_test_utils as lktu

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})


def test_poisson_basic_build():
    algo = poisson.HPF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None


def test_poisson_predict_basic():
    algo = poisson.HPF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None

    preds = algo.predict(model, 10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= 0
    assert preds.loc[3]


def test_poisson_predict_bad_item():
    algo = poisson.HPF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None

    preds = algo.predict(model, 10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_poisson_predict_bad_user():
    algo = poisson.HPF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None

    preds = algo.predict(model, 50, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


@mark.slow
def test_poisson_train_large():
    algo = poisson.HPF(20, iterations=20)
    ratings = lktu.ml_pandas.renamed.ratings
    model = algo.train(ratings)

    assert model is not None
    assert model.n_users == ratings.user.nunique()
    assert model.n_items == ratings.item.nunique()


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_poisson_batch_accuracy():
    import lenskit.crossfold as xf
    from lenskit import batch, topn
    import lenskit.metrics.topn as lm

    ratings = lktu.ml100k.load_ratings()

    algo = poisson.HPF(25, iterations=20)

    def eval(train, test):
        _log.info('running training')
        train['rating'] = train.rating.astype(np.float_)
        model = algo.train(train)
        users = test.user.unique()
        _log.info('testing %d users', len(users))
        candidates = topn.UnratedCandidates(train)
        recs = batch.recommend(algo, model, users, 100, candidates, test)
        return recs

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    recs = pd.concat(eval(train, test) for (train, test) in folds)

    _log.info('analyzing recommendations')
    ndcg = recs.groupby('user').rating.apply(lm.ndcg)
    _log.info('ndcg for users is %.4f', ndcg)
    assert ndcg.mean() > 0
