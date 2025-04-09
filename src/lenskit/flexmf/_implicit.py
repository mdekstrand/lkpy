# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import torch
from pydantic import model_validator
from torch.nn import functional as F

from lenskit.data import Dataset

from ._base import FlexMFConfigBase, FlexMFScorerBase
from ._model import FlexMFModel
from ._training import FlexMFTrainerBase, FlexMFTrainingBatch, FlexMFTrainingData

ImplicitLoss: TypeAlias = Literal["logistic", "pairwise", "warp"]
NegativeStrategy: TypeAlias = Literal["uniform", "popular", "misranked"]


@dataclass
class FlexMFImplicitConfig(FlexMFConfigBase):
    """
    Configuration for :class:`FlexMFImplicitScorer`.  It inherits base model
    options from :class:`FlexMFConfigBase`.

    Stability:
        Experimental
    """

    loss: ImplicitLoss = "logistic"
    """
    The loss to use for model training.
    """

    negative_strategy: NegativeStrategy | None = None
    """
    The negative sampling strategy.  The default is ``"misranked"`` for WARP
    loss and ``"uniform"`` for other losses.
    """

    negative_count: int = 1
    """
    The number of negative items to sample for each positive item in the
    training data.  With BPR loss, the positive item is compared to each
    negative item; with logistic loss, the positive item is treated once per
    learning round, so this setting effectively makes the model learn on _n_
    negatives per positive, rather than giving positive and negative examples
    equal weight.
    """

    positive_weight: float = 1.0
    """
    A weighting multiplier to apply to the positive item's loss, to adjust the
    relative importance of positive and negative classifications.  Only applies
    to logistic loss.
    """

    user_bias: bool | None = None
    """
    Whether to learn a user bias term.  If unspecified, the default depends on
    the loss function (``False`` for pairwise and ``True`` for logistic).
    """
    item_bias: bool = True
    """
    Whether to learn an item bias term.
    """

    def selected_negative_strategy(self) -> NegativeStrategy:
        if self.negative_strategy is not None:
            return self.negative_strategy
        elif self.loss == "warp":
            return "misranked"
        else:
            return "uniform"

    @model_validator(mode="after")
    def check_strategies(self):
        if (
            self.loss == "warp"
            and self.negative_strategy is not None
            and self.negative_strategy != "misranked"
        ):
            raise ValueError("WARP loss requires “misranked” negative strategy")

        if self.selected_negative_strategy() and self.negative_count > 1:
            raise ValueError("misrank negatives only works with single negatives")

        return self


class FlexMFImplicitScorer(FlexMFScorerBase):
    """
    Implicit-feedback rating prediction with FlexMF.  This is capable of
    realizing multiple models, including:

    - BPR-MF (Bayesian personalized ranking) :cite:p:`BPR` (with ``"pairwise"`` loss)
    - Logistic matrix factorization :cite:p:`LogisticMF` (with ``"logistic"`` loss)

    All use configurable negative sampling, including the sampling approach from WARP.

    Stability:
        Experimental
    """

    config: FlexMFImplicitConfig

    def create_trainer(self, data, options):
        if self.config.selected_negative_strategy() == "misranked":
            return FlexMFWARPTrainer(self, data, options)
        else:
            return FlexMFImplicitTrainer(self, data, options)


class FlexMFImplicitTrainer(FlexMFTrainerBase[FlexMFImplicitScorer, FlexMFImplicitConfig]):
    def prepare_data(self, data: Dataset) -> FlexMFTrainingData:
        """
        Set up the training data and context for the scorer.
        """

        matrix = data.interactions().matrix()
        coo = matrix.coo_structure()

        # save data we learned at this stage
        self.component.users = data.users
        self.component.items = data.items

        return FlexMFTrainingData(
            batch_size=self.config.batch_size,
            n_users=data.user_count,
            n_items=data.item_count,
            users=coo.row_numbers,
            items=coo.col_numbers,
            matrix=matrix,
        )

    def create_model(self) -> FlexMFModel:
        """
        Prepare the model for training.
        """
        user_bias = self.config.user_bias
        if user_bias is None:
            if self.config.loss == "pairwise":
                user_bias = False
            else:
                user_bias = True

        return FlexMFModel(
            self.config.embedding_size,
            self.data.n_users,
            self.data.n_items,
            self.torch_rng,
            user_bias=user_bias,
            item_bias=self.config.item_bias,
            sparse=self.config.reg_method != "AdamW",
        )

    def score(self, users, items) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.reg_method == "L2":
            result = self.model(users, items, return_norm=True)
            scores = result[0, ...]

            norms = result[1, ...]
        else:
            scores = self.model(users, items)
            norms = torch.tensor(0.0)

        return scores, norms

    def train_batch(self, batch: FlexMFTrainingBatch) -> float:
        users = torch.as_tensor(batch.users.reshape(-1, 1)).to(self.device)
        positives = torch.as_tensor(batch.items.reshape(-1, 1)).to(self.device)

        pos_pred, pos_norm = self.score(users, positives)

        neg_items, neg_pred, neg_norm, weights = self.scored_negatives(batch, users, pos_pred)

        match self.config.loss:
            case "logistic":
                pos_lp = -F.logsigmoid(pos_pred) * self.config.positive_weight
                neg_lp = -F.logsigmoid(-neg_pred)
                tot_lp = pos_lp.sum() + neg_lp.sum()
                tot_n = pos_lp.nelement() + neg_lp.nelement()
                loss = tot_lp / tot_n

            case "pairwise":
                lp = -F.logsigmoid(pos_pred - neg_pred)
                loss = lp.mean()
            case "warp":
                assert weights is not None
                lp = -F.logsigmoid(pos_pred - neg_pred) * weights
                loss = lp.mean()

        loss_all = loss + (pos_norm + neg_norm) * 0.5

        loss_all.backward()
        self.opt.step()

        return loss.item()

    def scored_negatives(
        self, batch: FlexMFTrainingBatch, users: torch.Tensor, pos_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert batch.data.matrix is not None
        assert isinstance(batch.users, np.ndarray)
        items = torch.as_tensor(
            batch.data.matrix.sample_negatives(
                batch.users,
                weighting=self.config.selected_negative_strategy(),
                n=self.config.negative_count,
                rng=self.rng,
            )
        ).to(users.device)
        scores, norms = self.score(users, items)
        return items, scores, norms, None


class FlexMFWARPTrainer(FlexMFImplicitTrainer):
    def scored_negatives(self, batch, users, pos_scores):
        assert batch.data.matrix is not None
        assert isinstance(batch.users, np.ndarray)

        ps = pos_scores.reshape(len(pos_scores), 1)

        # start with 100 negatives to find positive
        neg_cand = torch.as_tensor(
            batch.data.matrix.sample_negatives(
                batch.users,
                weighting="uniform",
                n=100,
                rng=self.rng,
            )
        )
        neg_cand = neg_cand.to(users.device)
        neg_scores, neg_norms = self.score(users, neg_cand)
        neg_misranked = neg_scores > ps
        neg_pos = torch.argmax(neg_misranked, dim=1)
        assert neg_pos.shape == (len(users),)

        uidx = torch.arange(len(users), device=users.device)
        n_not_mr = torch.sum(torch.logical_not(neg_misranked[uidx, neg_pos]))
        if n_not_mr:
            self.log.warning("%d items are not misranked", n_not_mr)

        ranks = (neg_pos + 1).to(torch.float64)
        # L(k) = sum i=1..k 1/i = harmonic k
        # approximate harmonic k with log
        weights = (
            torch.log(ranks)
            + np.euler_gamma
            + 1 / (2 * ranks)
            - 1 / (12 * ranks**2)
            + 1 / (120 * ranks**4)
        )
        return neg_cand[uidx, neg_pos], neg_scores[uidx, neg_pos], neg_norms[uidx, neg_pos], weights
