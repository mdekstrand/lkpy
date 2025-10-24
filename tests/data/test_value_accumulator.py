# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np

import hypothesis.strategies as st
from hypothesis import given
from pytest import approx

from lenskit.data.accum import ValueAccumulator


def test_collect_empty():
    acc = ValueAccumulator()
    rv = acc.accumulate()

    assert rv["n"] == 0
    assert np.isnan(rv["mean"])
    assert np.isnan(rv["median"])
    assert np.isnan(rv["std"])


@given(st.floats(allow_infinity=False, allow_nan=False))
def test_collect_one(x):
    acc = ValueAccumulator()
    acc.add(x)
    rv = acc.accumulate()

    assert rv["n"] == 1
    assert rv["mean"] == x
    assert rv["median"] == x
    assert rv["std"] == approx(0.0)


@given(st.lists(st.floats(allow_infinity=False, allow_nan=False), min_size=2))
def test_collect_list(xs):
    acc = ValueAccumulator()
    for x in xs:
        acc.add(x)
    rv = acc.accumulate()

    assert rv["n"] == len(xs)
    assert rv["mean"] == approx(np.mean(xs))
    assert rv["median"] == approx(np.median(xs))
    assert rv["std"] == approx(np.std(xs), nan_ok=True)
