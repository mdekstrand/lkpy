# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Test utilities for LKPY tests.
"""

import os
import os.path
from contextlib import contextmanager

import numpy as np
import scipy.sparse as sps

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
import pytest
from hypothesis import assume

from lenskit.algorithms.basic import PopScore
from lenskit.algorithms.ranking import PlackettLuce
from lenskit.batch import recommend
from lenskit.crossfold import simple_test_pair
from lenskit.data.matrix import torch_sparse_from_scipy
from lenskit.datasets import ML100K, MovieLens

ml_test = MovieLens("data/ml-latest-small")
ml100k = ML100K("data/ml-100k")


@pytest.fixture(scope="session")
def demo_recs():
    """
    A demo set of train, test, and recommendation data.
    """
    train, test = simple_test_pair(ml_test.ratings, f_rates=0.5)

    users = test["user"].unique()
    algo = PopScore()
    algo = PlackettLuce(algo, rng_spec="user")
    algo.fit(train)

    recs = recommend(algo, users, 500)
    return train, test, recs


@contextmanager
def set_env_var(var, val):
    "Set an environment variable & restore it."
    old_val = os.environ.get(var, None)
    try:
        if val is None:
            if old_val is not None:
                del os.environ[var]
        else:
            os.environ[var] = val
        yield
    finally:
        if old_val is not None:
            os.environ[var] = old_val
        elif val is not None:
            del os.environ[var]


@st.composite
def coo_arrays(
    draw,
    shape=None,
    dtype=nph.floating_dtypes(endianness="=", sizes=[32, 64]),
    elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False, width=32),
) -> sps.coo_array:
    if shape is None:
        shape = st.tuples(st.integers(1, 100), st.integers(1, 100))

    if isinstance(shape, st.SearchStrategy):
        shape = draw(shape)

    if not isinstance(shape, tuple):
        shape = shape, shape
    rows, cols = shape
    if isinstance(rows, st.SearchStrategy):
        rows = draw(rows)
    if isinstance(cols, st.SearchStrategy):
        cols = draw(cols)

    mask = draw(nph.arrays(np.bool_, (rows, cols)))
    # at least one nonzero value
    assume(np.any(mask))
    nnz = int(np.sum(mask))

    ris, cis = np.nonzero(mask)

    vs = draw(
        nph.arrays(dtype, nnz, elements=elements),
    )

    return sps.coo_array((vs, (ris, cis)), shape=(rows, cols))


@st.composite
def sparse_arrays(draw, *, layout="csr", **kwargs):
    if isinstance(layout, list):
        layout = st.sampled_from(layout)
    if isinstance(layout, st.SearchStrategy):
        layout = draw(layout)

    M: sps.coo_array = draw(coo_arrays(**kwargs))

    match layout:
        case "csr":
            return M.tocsr()
        case "csc":
            return M.tocsc()
        case "coo":
            return M
        case _:
            raise ValueError(f"invalid layout {layout}")


@st.composite
def sparse_tensors(draw, *, layout="csr", **kwargs):
    if isinstance(layout, list):
        layout = st.sampled_from(layout)
    if isinstance(layout, st.SearchStrategy):
        layout = draw(layout)

    M: sps.coo_array = draw(coo_arrays(**kwargs))
    return torch_sparse_from_scipy(M, layout)  # type: ignore


jit_enabled = True
if "NUMBA_DISABLE_JIT" in os.environ:
    jit_enabled = False
if os.environ.get("PYTORCH_JIT", None) == "0":
    jit_enabled = False

wantjit = pytest.mark.skipif(not jit_enabled, reason="JIT required")
