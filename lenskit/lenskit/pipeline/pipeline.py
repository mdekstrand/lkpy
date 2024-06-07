# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

from typing import TypeVar

from .module import Module
from .port import InputPort, OutputPort

D = TypeVar("D")
"""
Type variable for data wired through a pipeline.
"""
M = TypeVar("M", bound=Module)


class Pipeline:
    """
    Pipeline abstraction.
    """

    def add_module(self, name: str, module: M) -> M:
        # TODO wire up the module
        return module

    def wire_port(self, output: OutputPort[D], input: InputPort[D]) -> None:
        # wire an input to an output
        raise NotImplementedError()
