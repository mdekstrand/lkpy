# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Arryw utility functions.
"""

from functools import partial

import numpy as np
import pyarrow as pa
import torch
from typing_extensions import Callable, Literal, TypeAlias, TypeVar, overload

from .mtarray import MTArray

A = TypeVar("A", bound=pa.Array, default=pa.Array)
Selector: TypeAlias = Callable[[MTArray | A | None], A | None]


def get_indexer(sel) -> Selector:
    """
    Get a selector that will apply the specified indexer.  This allows
    one indexer to be applied to multiple arrays.
    """
    if np.isscalar(sel):
        sel = pa.array([sel], pa.int32())  # type: ignore
        return partial(arrow_take, sel)
    elif isinstance(sel, slice):
        return partial(arrow_slice, sel)
    elif not isinstance(sel, pa.Array):
        sel = pa.array(sel)

    if pa.types.is_integer(sel.type):
        return partial(arrow_take, sel)
    elif pa.types.is_boolean(sel.type):
        return partial(arrow_filter, sel)
    else:  # pragma: nocover
        raise TypeError(f"invalid selector: {sel}")


def arrow_to_format(array: pa.Array, format: Literal["arrow", "numpy", "torch"]):
    """
    Convert an Arrow array into another format.

    Stability:
        Internal
    """
    match format:
        case "arrow":
            return array
        case "numpy":
            return array.to_numpy(zero_copy_only=False)
        case "torch":
            return torch.as_tensor(array.to_numpy(zero_copy_only=False, writable=True))
        case _:  # pragma: nocover
            raise ValueError(f"unknown format {format}")


@overload
def arrow_slice(sel: slice, array: MTArray | A) -> A: ...
@overload
def arrow_slice(sel: slice, array: MTArray | A | None) -> A | None: ...
def arrow_slice(sel: slice, array: MTArray | A | None) -> A | None:
    """
    Slice an Arrow array.
    """
    if array is None:
        return None
    elif isinstance(array, MTArray):
        array = array.arrow()

    if sel.step and sel.step != 1:
        raise ValueError("slices with steps unsupported")
    start = sel.start or 0
    if sel.stop is None:
        slen = len(array) - start
    elif sel.stop < 0:
        slen = len(array) - start + sel.stop
    else:
        slen = sel.stop - start

    return array.slice(start, slen)


@overload
def arrow_take(sel: pa.Int32Array, array: MTArray | A) -> A: ...
@overload
def arrow_take(sel: pa.Int32Array, array: MTArray | A | None) -> A | None: ...
def arrow_take(sel: pa.Int32Array, array: MTArray | A | None) -> A | None:
    """
    Select from an Arrow array by integer indices.
    """

    if array is None:
        return None
    elif isinstance(array, MTArray):
        array = array.arrow()

    return array.take(sel)


@overload
def arrow_filter(sel: pa.BooleanArray, array: MTArray | A) -> A: ...
@overload
def arrow_filter(sel: pa.BooleanArray, array: MTArray | A | None) -> A | None: ...
def arrow_filter(sel: pa.BooleanArray, array: MTArray | A | None) -> A | None:
    """
    Select from an Arrow array by integer indices.
    """

    if array is None:
        return None
    elif isinstance(array, MTArray):
        array = array.arrow()

    return array.filter(sel)


def arrow_type(dtype: np.dtype) -> pa.DataType:
    """
    Resolve a NumPy data type to an Arrow data type, assuming objects store
    strings.
    """
    if dtype == np.object_:
        return pa.utf8()
    else:
        return pa.from_numpy_dtype(dtype)
