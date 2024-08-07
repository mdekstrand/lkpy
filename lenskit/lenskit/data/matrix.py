# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Data manipulation routines.
"""

# pyright: basic
from __future__ import annotations

import logging
import platform

import numpy as np
import scipy.sparse as sps
import torch
from numpy.typing import ArrayLike
from typing_extensions import Literal, NamedTuple, Optional, TypeVar, overload

_log = logging.getLogger(__name__)

t = torch
M = TypeVar("M", "CSRStructure", sps.csr_array, sps.coo_array, sps.spmatrix, t.Tensor)


class CSRStructure(NamedTuple):
    """
    Representation of the compressed sparse row structure of a sparse matrix,
    without any data values.
    """

    rowptrs: np.ndarray
    colinds: np.ndarray
    shape: tuple[int, int]

    @property
    def nrows(self):
        return self.shape[0]

    @property
    def ncols(self):
        return self.shape[1]

    @property
    def nnz(self):
        return self.rowptrs[self.nrows]

    def extent(self, row: int) -> tuple[int, int]:
        return self.rowptrs[row], self.rowptrs[row + 1]

    def row_cs(self, row: int) -> np.ndarray:
        sp, ep = self.extent(row)
        return self.colinds[sp:ep]


class InteractionMatrix:
    """
    Internal helper class used by :class:`lenskit.data.Dataset` to store the
    user-item interaction matrix.  The data is stored simultaneously in CSR and
    COO format.  Most code has no need to interact with this class directly —
    :class:`~lenskit.data.Dataset` methods provide data in a range of formats.
    """

    n_obs: int
    n_users: int
    n_items: int

    user_nums: np.ndarray[int, np.dtype[np.int32]]
    "User (row) numbers."
    user_ptrs: np.ndarray[int, np.dtype[np.int32]]
    "User (row) offsets / pointers."
    item_nums: np.ndarray[int, np.dtype[np.int32]]
    "Item (column) numbers."
    ratings: Optional[np.ndarray[int, np.dtype[np.float32]]] = None
    "Rating values."
    timestamps: Optional[np.ndarray[int, np.dtype[np.int64]]] = None
    "Timestamps as 64-bit Unix timestamps."

    def __init__(
        self,
        users: ArrayLike,
        items: ArrayLike,
        ratings: Optional[ArrayLike],
        timestamps: Optional[ArrayLike],
        n_users: int,
        n_items: int,
    ):
        self.user_nums = np.asarray(users, np.int32)
        assert np.all(np.diff(self.user_nums) >= 0), "matrix data not sorted"
        self.item_nums = np.asarray(items, np.int32)
        if ratings is not None:
            self.ratings = np.asarray(ratings, np.float32)
        if timestamps is not None:
            self.timestamps = np.asarray(timestamps, np.int64)

        self.n_obs = len(self.user_nums)
        self.n_items = n_items
        self.n_users = n_users
        cp1 = np.zeros(self.n_users + 1, np.int32)
        np.add.at(cp1[1:], self.user_nums, 1)
        self.user_ptrs = cp1.cumsum(dtype=np.int32)
        if self.user_ptrs[-1] != len(self.user_nums):
            raise ValueError("mismatched counts and array sizes")

    @property
    def shape(self) -> tuple[int, int]:
        """
        The shape of the interaction matrix (rows x columns).
        """
        return (self.n_users, self.n_items)


@overload
def normalize_sparse_rows(
    matrix: t.Tensor, method: Literal["center"], inplace: bool = False
) -> tuple[t.Tensor, t.Tensor]: ...
@overload
def normalize_sparse_rows(
    matrix: t.Tensor, method: Literal["unit"], inplace: bool = False
) -> tuple[t.Tensor, t.Tensor]: ...
def normalize_sparse_rows(
    matrix: t.Tensor, method: str, inplace: bool = False
) -> tuple[t.Tensor, t.Tensor]:
    """
    Normalize the rows of a sparse matrix.
    """
    match method:
        case "unit":
            return _nsr_unit(matrix)
        case "center":
            return _nsr_mean_center(matrix)
        case _:
            raise ValueError(f"unsupported normalization method {method}")


def _nsr_mean_center(matrix: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    nr, _nc = matrix.shape
    sums = matrix.sum(dim=1, keepdim=True).to_dense().reshape(nr)
    counts = torch.diff(matrix.crow_indices())
    assert sums.shape == counts.shape
    means = torch.nan_to_num(sums / counts, 0)
    return t.sparse_csr_tensor(
        crow_indices=matrix.crow_indices(),
        col_indices=matrix.col_indices(),
        values=matrix.values() - t.repeat_interleave(means, counts),
        size=matrix.shape,
    ), means


def _nsr_unit(matrix: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    sqmat = t.sparse_csr_tensor(
        crow_indices=matrix.crow_indices(),
        col_indices=matrix.col_indices(),
        values=matrix.values().square(),
    )
    norms = sqmat.sum(dim=1, keepdim=True).to_dense().reshape(matrix.shape[0])
    norms.sqrt_()
    recip_norms = t.where(norms > 0, t.reciprocal(norms), 0.0)
    return t.sparse_csr_tensor(
        crow_indices=matrix.crow_indices(),
        col_indices=matrix.col_indices(),
        values=matrix.values() * t.repeat_interleave(recip_norms, matrix.crow_indices().diff()),
        size=matrix.shape,
    ), norms


def torch_sparse_from_scipy(
    M: sps.coo_array, layout: Literal["csr", "coo", "csc"] = "coo"
) -> t.Tensor:
    """
    Convert a SciPy :class:`sps.coo_array` into a torch sparse tensor.
    """
    ris = t.from_numpy(M.row)
    cis = t.from_numpy(M.col)
    vs = t.from_numpy(M.data)
    indices = t.stack([ris, cis])
    assert indices.shape == (2, M.nnz)
    T = t.sparse_coo_tensor(indices, vs, size=M.shape)
    assert T.shape == M.shape

    match layout:
        case "csr":
            return T.to_sparse_csr()
        case "csc":
            return T.to_sparse_csc()
        case "coo":
            return T.coalesce()
        case _:
            raise ValueError(f"invalid layout {layout}")


if platform.machine() == "arm64":

    @torch.jit.ignore  # type: ignore
    def safe_spmv(matrix, vector):  # type: ignore
        """
        Sparse matrix-vector multiplication working around PyTorch bugs.

        This is equivalent to :func:`torch.mv` for sparse CSR matrix
        and dense vector, but it works around PyTorch bug 127491_ by
        falling back to SciPy on ARM.

        .. _127491: https://github.com/pytorch/pytorch/issues/127491
        """
        assert matrix.is_sparse_csr
        nr, nc = matrix.shape
        M = sps.csr_array(
            (matrix.values().numpy(), matrix.col_indices().numpy(), matrix.crow_indices().numpy()),
            (nr, nc),
        )
        v = vector.numpy()
        return torch.from_numpy(M @ v)

else:

    def safe_spmv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """
        Sparse matrix-vector multiplication working around PyTorch bugs.

        This is equivalent to :func:`torch.mv` for sparse CSR matrix
        and dense vector, but it works around PyTorch bug 127491_ by
        falling back to SciPy on ARM.

        .. _127491: https://github.com/pytorch/pytorch/issues/127491
        """
        assert matrix.is_sparse_csr
        return torch.mv(matrix, vector)
