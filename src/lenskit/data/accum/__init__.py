# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Data accumulation support
"""

from ._proto import Accumulator, R, SplittableAccumulator, X

__all__ = [
    "Accumulator",
    "SplittableAccumulator",
    "X",
    "R",
]
