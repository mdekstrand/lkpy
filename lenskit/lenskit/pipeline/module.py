# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

from abc import ABC, abstractmethod


class Module(ABC):
    """
    Base class for pipeline modules.
    """

    # TODO better default name
    name: str = __name__
