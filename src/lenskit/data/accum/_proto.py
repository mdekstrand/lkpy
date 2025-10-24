# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing_extensions import Generic, Protocol, Self, TypeVar, runtime_checkable

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


@runtime_checkable
class SplittableAccumulator(Accumulator[X, R], Protocol):
    r"""
    Extension of :class:`Accumulator` for accumulators that support combining
    sub-accumulators.

    The accumulations and combination defined by such an accumulator must be
    **associative**, but they do not need to be commutative.
    """

    def split_accumulator(self) -> Self:
        """
        Split this accumulator into two. The currently-accumulated values, if
        any, are kept on `self`, and a new empty accumulator is returned.  If
        the accumulator is not commutative, then ``self`` is the left
        accumulator and method returns the right accumulator.

        The accumulators can later be combined with :meth:`combine`.

        Returns:
            The right-hand side of the split accumulator.
        """
        ...

    def combine(self, right: Self) -> None:
        """
        Combine another accumulator into this one.

        Args:
            right:
                The right-hand accumulator to accumulate into ``self`` (which
                is the left accumulator).
        """
        ...
