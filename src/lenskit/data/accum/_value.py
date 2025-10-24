# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import TypedDict

import numpy as np

from ._proto import Accumulator


class ValueStatistics(TypedDict):
    """
    Collected statitsics from :class:`ValueAccumulator`.
    """

    mean: float
    median: float
    std: float


class ValueAccumulator(Accumulator[float, dict[str, float]]):
    """
    An accumulator for single real values, computing basic statistics.
    """

    _values: list[float]

    def __init__(self):
        self._values = []

    def add(self, value):
        self._values.append(value)

    def accumulate(self) -> ValueStatistics:
        return {
            "mean": np.mean(self._values).item(),
            "median": np.median(self._values).item(),
            "std": np.std(self._values).item(),
        }
