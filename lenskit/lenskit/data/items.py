# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Primary item-list abstraction.
"""

from __future__ import annotations

from typing import Any, Literal, LiteralString, Sequence, TypeAlias, TypeVar, cast, overload

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike, NDArray

from lenskit.data.checks import check_1d
from lenskit.data.mtarray import MTArray, MTGenericArray
from lenskit.data.vocab import EntityId, NPEntityId, Vocabulary

Backend: TypeAlias = Literal["numpy", "torch"]
EID = TypeVar("EID", bound=EntityId)


class ItemList:
    """
    Representation of a (usually ordered) list of items, possibly with scores
    and other associated data; many components take and return item lists.  Item
    lists are to be treated as **immutable** — create a new list with modified
    data, do not do in-place modifications of the list itself or the arrays or
    data frame it returns.

    An item list logically a list of rows, each of which is an item, like a
    :class:`~pandas.DataFrame` but supporting multiple array backends.

    .. note::

        Naming for fields and accessor methods is tricky, because the usual
        convention for a data frame is to use singular column names (e.g.
        “item_id”, “score”) instead of plural (“item_ids”, “scores”) — the data
        frame, like a database table, is a list of instances, and the column
        names are best interpreted as naming attributes of individual instances.

        However, when working with a list of e.g. item IDs, it is more natural —
        at least to this author — to use plural names: ``item_ids``.  Since this
        class is doing somewhat double-duty, representing a list of items along
        with associated data, as well as a data frame of columns representing
        items, the appropriate naming is not entirely clear.  The naming
        convention in this class is therefore as follows:

        * Field names are singular (``item_id``, ``score``).
        * Named accessor methods are plural (:meth:`item_ids`, :meth:`scores`).
        * Both singular and plural forms are accepted for item IDs numbers, and
          scores in the keyword arguments.  Other field names should be
          singular.

    Args:
        item_ids:
            A list or array of item identifiers. ``item_id`` is accepted as an
            alternate name.
        item_nums:
            A list or array of item numbers. ``item_num`` is accepted as an
            alternate name.
        vocabulary:
            A vocabulary to translate between item IDs and numbers.
        ordered:
            Whether the list has a meaningful order.
        scores:
            An array of scores for the items.
        fields:
            Additional fields, such as ``score`` or ``rating``.  Field names
            should generally be singular; the named keyword arguments and
            accessor methods are plural for readability (“get the list of item
            IDs”)
    """

    ordered: bool
    "Whether this list has a meaningful order."
    _len: int
    _ids: np.ndarray[int, np.dtype[NPEntityId]] | None = None
    _numbers: MTArray[np.int32] | None = None
    _vocab: Vocabulary[EntityId] | None = None
    _ranks: MTArray[np.int32] | None = None
    _fields: dict[str, MTGenericArray]

    def __init__(
        self,
        *,
        item_ids: NDArray[NPEntityId] | pd.Series[EntityId] | Sequence[EntityId] | None = None,
        item_nums: NDArray[np.int32] | pd.Series[int] | Sequence[int] | ArrayLike | None = None,
        vocabulary: Vocabulary[EID] | None = None,
        ordered: bool = False,
        scores: NDArray[np.generic] | torch.Tensor | ArrayLike | None = None,
        **fields: NDArray[np.generic] | torch.Tensor | ArrayLike,
    ):
        self.ordered = ordered
        self._vocab = vocabulary

        if item_ids is None and "item_id" in fields:
            item_ids = np.asarray(cast(Any, fields["item_id"]))

        if item_nums is None and "item_num" in fields:
            item_nums = np.asarray(cast(Any, fields["item_num"]))
            if not issubclass(item_nums.dtype.type, np.integer):
                raise TypeError("item numbers not integers")

        if item_ids is None and item_nums is None:
            self._ids = np.ndarray(0, dtype=np.int32)
            self._numbers = MTArray(np.ndarray(0, dtype=np.int32))
            self._len = 0

        if item_ids is not None:
            self._ids = np.asarray(item_ids)
            if not issubclass(self._ids.dtype.type, (np.integer, np.str_, np.bytes_)):
                raise TypeError(f"item IDs not integers or bytes (type: {self._ids.dtype})")

            check_1d(self._ids, label="item_ids")
            self._len = len(item_ids)

        if item_nums is not None:
            self._numbers = MTArray(item_nums)
            check_1d(self._numbers, getattr(self, "_len", None), label="item_nums")
            self._len = self._numbers.shape[0]

        # convert fields and drop singular ID/number aliases
        self._fields = {
            name: check_1d(MTArray(data), self._len, label=name)
            for (name, data) in fields.items()
            if name not in ("item_id", "item_num")
        }

        if scores is not None:
            if "score" in fields:  # pragma: nocover
                raise ValueError("cannot specify both scores= and score=")
            self._fields["score"] = MTArray(scores)

    def clone(self) -> ItemList:
        """
        Make a shallow copy of the item list.
        """
        return ItemList(
            item_ids=self._ids,
            item_nums=self._numbers,
            vocabulary=self._vocab,
            ordered=self.ordered,
            **self._fields,
        )

    def ids(self) -> NDArray[NPEntityId]:
        """
        Get the item IDs.

        Returns:
            An array of item identifiers.

        Raises:
            RuntimeError: if the item list was not created with IDs or a :class:`Vocabulary`.
        """
        if self._ids is None:
            if self._vocab is None:
                raise RuntimeError("item IDs not available (no IDs or vocabulary provided)")
            assert self._numbers is not None
            self._ids = self._vocab.ids(self._numbers.numpy())

        return self._ids

    @overload
    def numbers(self, format: Literal["numpy"] = "numpy") -> NDArray[np.int32]: ...
    @overload
    def numbers(self, format: Literal["torch"]) -> torch.Tensor: ...
    @overload
    def numbers(self, format: LiteralString = "numpy") -> ArrayLike: ...
    def numbers(self, format: LiteralString = "numpy") -> ArrayLike:
        """
        Get the item numbers.

        Args:
            format:
                The array format to use.

        Returns:
            An array of item numbers.

        Raises:
            RuntimeError: if the item list was not created with numbers or a :class:`Vocabulary`.
        """
        if self._numbers is None:
            if self._vocab is None:
                raise RuntimeError("item numbers not available (no IDs or vocabulary provided)")
            assert self._ids is not None
            self._numbers = MTArray(self._vocab.numbers(self._ids))

        return self._numbers.to(format)

    @overload
    def scores(self, format: Literal["numpy"] = "numpy") -> NDArray[np.floating] | None: ...
    @overload
    def scores(self, format: Literal["torch"]) -> torch.Tensor | None: ...
    @overload
    def scores(self, format: LiteralString = "numpy") -> ArrayLike | None: ...
    def scores(self, format: LiteralString = "numpy") -> ArrayLike | None:
        """
        Get the item scores (if available).
        """
        return self.field("score", format)

    @overload
    def ranks(self, format: Literal["numpy"] = "numpy") -> NDArray[np.int32] | None: ...
    @overload
    def ranks(self, format: Literal["torch"]) -> torch.Tensor | None: ...
    @overload
    def ranks(self, format: LiteralString = "numpy") -> ArrayLike | None: ...
    def ranks(self, format: LiteralString = "numpy") -> ArrayLike | None:
        """
        Get an array of ranks for the items in this list, if it is ordered.
        Unordered lists have no ranks.  The ranks are based on the order in the
        list, **not** on the score.

        Item ranks start with **1**, for compatibility with common practice in
        mathematically defining information retrieval metrics and operations.

        Returns:
            An array of item ranks, or ``None`` if the list is unordered.
        """
        if not self.ordered:
            return None

        if self._ranks is None:
            self._ranks = MTArray(np.arange(1, self._len + 1, dtype=np.int32))

        return self._ranks.to(format)

    @overload
    def field(
        self, name: str, format: Literal["numpy"] = "numpy"
    ) -> NDArray[np.floating] | None: ...
    @overload
    def field(self, name: str, format: Literal["torch"]) -> torch.Tensor | None: ...
    @overload
    def field(self, name: str, format: LiteralString) -> ArrayLike | None: ...
    def field(self, name: str, format: LiteralString = "numpy") -> ArrayLike | None:
        val = self._fields.get(name, None)
        if val is None:
            return None
        else:
            return val.to(format)

    def __len__(self):
        return self._len
