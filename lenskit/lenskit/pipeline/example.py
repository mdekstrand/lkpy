"""
Example of using the pipeline abstraction.
"""

# pyright: strict
from dataclasses import dataclass, field

import pandas as pd
import torch
from typing_extensions import (
    Any,
    Callable,
    Generic,
    NamedTuple,
    NewType,
    ParamSpec,
    Self,
    TypeAlias,
    TypedDict,
    TypeVar,
    Unpack,
    cast,
    dataclass_transform,
    reveal_type,
)

from .module import Module
from .pipeline import Pipeline
from .port import RuntimeInput, TrainingInput, TrainingOutput

UIFeedback = NewType("UIFeedback", pd.DataFrame)
# need more flexibility here
User = NewType("User", int)
# this type is kinda bad
ItemList = NewType("ItemList", list[int])


class InputsBase(TypedDict):
    pass


class PredictInputs(InputsBase):
    user: int


IV = TypeVar("IV", bound=InputsBase)


class Component(Generic[IV]):
    pass


class Predictor(Component[PredictInputs]):
    pass


def wire(call: Component[IV], **kwargs: Unpack[IV]):
    pass


def predict(*, user: int):
    pass


wire(Predictor(), user=134)

wire(Predictor(), user="bob")
