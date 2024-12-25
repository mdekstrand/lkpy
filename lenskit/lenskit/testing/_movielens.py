"""
MovieLens test fixtures and data marks.
"""

from typing import Generator

import numpy as np
import pandas as pd
import structlog
from pyprojroot import here

import pytest

from lenskit.basic import PopScorer, SoftmaxRanker
from lenskit.batch import recommend
from lenskit.data import Dataset, ItemListCollection, UserIDKey, from_interactions_df
from lenskit.data.lazy import LazyDataset
from lenskit.data.movielens import load_movielens, load_movielens_df
from lenskit.pipeline import RecPipelineBuilder
from lenskit.splitting import TTSplit, simple_test_pair

_log = structlog.stdlib.get_logger("lenskit.testing")

ml_test_dir = here("data/ml-latest-small")
ml_100k_zip = here("data/ml-100k.zip")

ml_test: Dataset = LazyDataset(lambda: load_movielens(ml_test_dir))
_ml_ds_cache: Dataset | None = None


@pytest.fixture(scope="session")
def ml_ratings() -> Generator[pd.DataFrame, None, None]:
    """
    Fixture to load the test MovieLens ratings as a data frame. To use this,
    just include it as a parameter in your test::

        def test_thing_with_data(ml_ratings: pd.DataFrame):
            ...

    .. note::
        This is imported in ``conftest.py`` so it is always available in LensKit tests.
    """
    yield load_movielens_df(ml_test_dir)


@pytest.fixture()
def ml_ds(ml_ratings: pd.DataFrame) -> Generator[Dataset, None, None]:
    """
    Fixture to load the MovieLens test dataset.  To use this, just include it as
    a parameter in your test::

        def test_thing_with_data(ml_ds: Dataset):
            ...

    .. note::
        This is imported in ``conftest.py`` so it is always available in LensKit tests.
    """
    global _ml_ds_cache
    log = _log.bind()
    ds = _ml_ds_cache
    if ds is None:
        _ml_ds_cache = ds = from_interactions_df(ml_ratings)

    old_rates = ds.interaction_matrix("pandas", field="rating", original_ids=True).copy(deep=True)
    old_ustats = ds.user_stats().copy(deep=True)
    old_istats = ds.item_stats().copy(deep=True)

    yield ds

    ustats = ds.user_stats()
    istats = ds.item_stats()

    rates = ds.interaction_matrix("pandas", field="rating", original_ids=True)
    assert rates["rating"].values == pytest.approx(old_rates["rating"].values)

    for col in old_ustats.columns:
        log.info("checking user stats column", column=col)
        assert ustats[col].values == pytest.approx(old_ustats[col].values)

    for col in old_istats.columns:
        log.info("checking item stats column", column=col)
        assert istats[col].values == pytest.approx(old_istats[col].values)


@pytest.fixture
def ml_100k() -> Generator[pd.DataFrame, None, None]:
    """
    Fixture to load the MovieLens 100K dataset (currently as a data frame).  It skips
    the test if the ML100K data is not available.
    """
    if not ml_100k_zip.exists():
        pytest.skip("ML100K data not available")
    yield load_movielens_df(ml_100k_zip)


@pytest.fixture(scope="session")
def demo_recs() -> tuple[TTSplit, ItemListCollection[UserIDKey]]:
    """
    A demo set of train, test, and recommendation data.
    """
    rng = np.random.default_rng(42)
    ml_ds = load_movielens(ml_test_dir)
    split = simple_test_pair(ml_ds, f_rates=0.5, rng=rng)

    builder = RecPipelineBuilder()
    builder.scorer(PopScorer())
    builder.ranker(SoftmaxRanker(500))
    pipe = builder.build()
    pipe.train(split.train)

    recs = recommend(pipe, list(split.test.keys()), 500, n_jobs=1, rng=rng)
    return split, recs
