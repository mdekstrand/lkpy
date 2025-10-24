# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import Generic, Protocol, TypeVar, runtime_checkable

X = TypeVar("X", contravariant=True)
R = TypeVar("R", covariant=True)


@runtime_checkable
class Accumulator(Protocol, Generic[X, R]):
    """
    Protocol implemented by data accumulators.
    """

    def add(self, value: X) -> None:
        """
        Add a single value to this accumulator.
        """
        ...

    def accumulate(self) -> R:
        """
        Compute the accumulated value from this accumulator.
        """
        ...
