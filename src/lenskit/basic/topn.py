# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Basic Top-*N* ranking.
"""

import logging

from pydantic import BaseModel

from lenskit.data import ItemList
from lenskit.pipeline.components import Component

_log = logging.getLogger(__name__)


class TopNConfig(BaseModel):
    """
    Configuration for top-N ranking.
    """

    n: int | None = None
    """
    The number of items to return. -1 or ``None`` to return all scored items.
    """


class TopNRanker(Component[ItemList]):
    """
    Rank scored items by their score and take the top *N*.  The ranking length
    can be passed either at runtime or at component instantiation time, with the
    latter taking precedence.

    Stability:
        Caller
    """

    config: TopNConfig
    "Configuration object."

    def __call__(self, *, items: ItemList, n: int | None = None) -> ItemList:
        """
        Rank the items.

        Args:
            items:
                The items to rank, with scores.  Items with missing scores are
                not included in the final ranking.
            n:
                The number of items to return, or -1 to return all scored items.
                If ``None``, the length configured at construction time is used.

        Returns:
            An ordered list of items, with scores and all other attributes
            preserved.
        """
        if n is None:
            n = self.config.n or -1

        if n >= 0:
            _log.debug("ranking top %d of %d items", n, len(items))
        else:
            _log.debug("ranking all of %d items", len(items))

        return items.top_n(n)
