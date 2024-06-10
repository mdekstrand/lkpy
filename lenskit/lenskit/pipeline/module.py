# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Self, TypeVar

R = TypeVar("R")


class Module(ABC, Generic[R]):
    """
    Base class for pipeline modules.
    """

    # TODO better default name
    name: str = __name__

    def fit(self, *args: Any, **kwargs: Any) -> Self:
        # default does nothing
        return self

    @abstractmethod
    def apply(self, *args: Any, **kwargs: Any) -> R: ...
