# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict

import warnings
from inspect import Signature, signature

from typing_extensions import Generic, TypeVar

from .components import PipelineFunction
from .types import TypecheckWarning

# Nodes are (conceptually) immutable data containers, so Node[U] can be assigned
# to Node[T] if U ≼ T.
ND = TypeVar("ND", covariant=True)


class Node(Generic[ND]):
    """
    Representation of a single node in a :class:`Pipeline`.

    Stability:
        Caller
    """

    __match_args__ = ("name",)

    name: str
    "The name of this node."
    types: set[type] | None
    "The set of valid data types of this node, or None for no typechecking."

    def __init__(self, name: str, *, types: set[type] | None = None):
        self.name = name
        self.types = types

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"


class InputNode(Node[ND], Generic[ND]):
    """
    An input node.

    Stability:
        Internal
    """


class LiteralNode(Node[ND], Generic[ND]):
    """
    A node storing a literal value.

    Stability:
        Internal
    """

    __match_args__ = ("name", "value")
    value: ND
    "The value associated with this node"

    def __init__(self, name: str, value: ND, *, types: set[type] | None = None):
        super().__init__(name, types=types)
        self.value = value


class ComponentNode(Node[ND], Generic[ND]):
    """
    A node storing a component.

    Stability:
        Internal
    """

    __match_args__ = ("name", "component", "inputs", "connections")

    component: PipelineFunction[ND]
    "The component associated with this node"

    inputs: dict[str, type | None]
    "The component's inputs."

    connections: dict[str, str]
    "The component's input connections."

    def __init__(self, name: str, component: PipelineFunction[ND]):
        super().__init__(name)
        self.component = component
        self.connections = {}

        sig = signature(component, eval_str=True)
        if sig.return_annotation == Signature.empty:
            warnings.warn(
                f"component {component} has no return type annotation", TypecheckWarning, 2
            )
        else:
            self.types = set([sig.return_annotation])

        self.inputs = {}
        for param in sig.parameters.values():
            if param.annotation == Signature.empty:
                warnings.warn(
                    f"parameter {param.name} of component {component} has no type annotation",
                    TypecheckWarning,
                    2,
                )
                self.inputs[param.name] = None
            else:
                self.inputs[param.name] = param.annotation
