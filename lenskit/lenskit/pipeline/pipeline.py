# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

from typing import Any, TypeVar, overload

from .module import Module
from .port import InputPort, OutputPort

D = TypeVar("D")
"""
Type variable for data wired through a pipeline.
"""
M = TypeVar("M", bound=Module[Any])


class Pipeline:
    """
    Pipeline abstraction.
    """

    def add_module(self, name: str, module: M) -> M:
        # TODO wire up the module
        return module

    @overload
    def wire_port(self, output: OutputPort[D], input: InputPort[D]) -> None: ...
    @overload
    def wire_port(self, output: Module[D], input: InputPort[D]) -> None: ...
    def wire_port(self, output: OutputPort[D] | Module[D], input: InputPort[D]) -> None:
        # wire an input to an output
        raise NotImplementedError()

    def fit(self, **data: Any):
        """
        Fit the modules in the pipeline.
        """
        # TODO: the signature for this method is not type-safe.
        # I do not know a way to fix that.

    def apply(self, mod: Module[D], **inputs: Any) -> D:
        # TODO: run the pipeline up through mod, passing inputs as needed
        ...
