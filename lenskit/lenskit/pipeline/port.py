# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar

# TODO verify that these covariances are correct
InData = TypeVar("InData", contravariant=True)
OutData = TypeVar("OutData", covariant=True)


class DataPort:
    """
    Data port for routing data between modules.  Ports can either be bound to a
    specific module instance or “unbound”.  They should be assigned, unbound, to
    class variables to “declare” the ports, and the :class:`Module` constructor
    converts unbound class variable ports into bound instance variable ports
    automatically.
    """

    name: str
    """
    The name of this port.
    """

    module_instance: Optional[Module] = None
    """
    The module instance to which this port is bound (after a module is
    instantiated).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def is_bound(self) -> bool:
        """
        Query whether this port is bound to a specific module instance.
        """
        return self.module_instance is not None


class InputPort(DataPort, Generic[InData]):
    """
    Data port for a module's input(s).
    """


class OutputPort(DataPort, Generic[OutData]):
    """
    A data port for a module's output(s).

    .. note::
        This is only used for training outputs — runtime outputs are just
        returned from the module's function(s).
    """


class TrainingInput(InputPort[InData], Generic[InData]):
    """
    Input that a module uses for its training process.
    """

    pass


class TrainingOutput(OutputPort[OutData], Generic[OutData]):
    """
    An output of a module's training process.
    """


class RuntimeInput(InputPort[InData], Generic[InData]):
    """
    Input that a module uses for its runtime computation (“inference”).
    """


from .module import Module  # noqa: E402
