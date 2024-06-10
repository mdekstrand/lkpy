"""
Example of using the pipeline abstraction.
"""

# pyright: strict
from typing import Any, NewType, Self, cast

import pandas as pd
import torch

from .module import Module
from .pipeline import Pipeline
from .port import RuntimeInput, TrainingInput, TrainingOutput

UIFeedback = NewType("UIFeedback", pd.DataFrame)
# need more flexibility here
User = NewType("User", int)
# this type is kinda bad
ItemList = NewType("ItemList", list[int])


class ALSEmbeddings(Module[None]):
    """
    Compute user-item embeddings with ALS.

    .. note::
        Its module data type is ``None`` because it does not, on its own, do anything
        at runtime.
    """

    ratings = TrainingInput[UIFeedback]("ratings")
    item_embeddings = TrainingOutput[torch.Tensor]("item_embeddings")
    user_embeddings = TrainingOutput[torch.Tensor]("user_embeddings")

    def fit(self, ratings: UIFeedback) -> Self:
        # question: take ratings as a parameter or obtain it from self.ratings?
        return self

    def apply(self, **kwargs: Any) -> None:
        return None


class EmbeddingInnerScore(Module[pd.Series[float]]):
    user = RuntimeInput[User]("user")
    items = RuntimeInput[ItemList]("items")
    item_embeddings = TrainingInput[torch.Tensor]("item_embeddings")
    user_embeddings = TrainingInput[torch.Tensor]("user_embeddings")

    def apply(
        self,
        user: User,
        items: ItemList,
        item_embeddings: torch.Tensor,
        user_embeddings: torch.Tensor,
    ) -> pd.Series[float]:
        # score items by dot products
        raise NotImplementedError()


class UnratedItemsCandidateSelector(Module[ItemList]):
    ratings = TrainingInput[UIFeedback]("ratings")
    user = RuntimeInput[User]("user")

    def fit(self, ratings: UIFeedback) -> Self:
        # memorize the ratings (since this is currentlyin a memorized-ratings model)
        return self

    def apply(self, **kwargs: Any) -> ItemList:
        # TODO actually return things
        return cast(ItemList, [])


class TopNRanker(Module[pd.Series[float]]):
    n: int
    item_scores = RuntimeInput[pd.Series[float]]("item_scores")

    def __init__(self, n: int):
        self.n = n

    def apply(self, item_scores: pd.Series[float]) -> pd.Series[float]:
        return item_scores.nlargest(self.n)


def wire_recommender():
    pipe = Pipeline()
    als = pipe.add_module("embed", ALSEmbeddings())
    score = pipe.add_module("score", EmbeddingInnerScore())
    select = pipe.add_module("select", UnratedItemsCandidateSelector())
    rank = pipe.add_module("select", TopNRanker(10))

    pipe.wire_port(als.item_embeddings, score.item_embeddings)
    pipe.wire_port(als.user_embeddings, score.user_embeddings)
    pipe.wire_port(select, score.items)
    pipe.wire_port(score, rank.item_scores)

    # what if I try to miswire? it should fail to type-check
    pipe.wire_port(als.item_embeddings, score.items)

    # open question: can we tell the pipeline what its last thing is in a type-safe way?
    # but we can run things here (method not yet implemented).
    # fit the model
    pipe.fit(ratings=...)
    # apply the model, providing the final stage & the user input
    recs = pipe.apply(rank, user=34128)
    # recs is now a series, you can check with hover!
