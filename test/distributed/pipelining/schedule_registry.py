# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# This file is a Schedule zoo for testing torch.distributed.pipelining.
# It includes schedules designed purely for testing purposes
from typing import Callable, Dict, List, Optional

from torch.distributed.pipelining.schedules import (
    _Action,
    _ComputationType,
    PipelineScheduleMulti,
)
from torch.distributed.pipelining.stage import _PipelineStageBase

F = _ComputationType.FORWARD
B = _ComputationType.BACKWARD
W = _ComputationType.WEIGHT


class ScheduleVShaped(PipelineScheduleMulti):
    n_stages = 4
    rank_stages = {
        0: [0, 3],
        1: [1, 2],
    }

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        stage_index_to_group_rank: Dict[int, int],
        loss_fn: Optional[Callable] = None,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            stage_index_to_group_rank=stage_index_to_group_rank,
        )

        # Go through one microbatch
        # F0_0 None None F0_3 B0_3 None None B0_0
        # None F0_1 F0_2 None None B0_2 B0_1 None
        self.pipeline_order = {
            0: [
                _Action(F, 0, 0),
                None,
                None,
                _Action(F, 0, 3),
                _Action(B, 0, 3),
                None,
                None,
                _Action(B, 0, 0),
            ],
            1: [
                None,
                _Action(F, 0, 1),
                _Action(F, 0, 2),
                None,
                None,
                _Action(B, 0, 2),
                _Action(B, 0, 1),
                None,
            ],
        }


class ScheduleUnbalanced(PipelineScheduleMulti):
    n_stages = 5
    rank_stages = {
        0: [0, 1, 4],
        1: [2, 3],
    }

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        stage_index_to_group_rank: Dict[int, int],
        loss_fn: Optional[Callable] = None,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            stage_index_to_group_rank=stage_index_to_group_rank,
        )

        self.pipeline_order = {
            0: [
                _Action(F, 0, 0),
                _Action(F, 0, 1),
                None,
                None,
                _Action(F, 0, 4),
                _Action(B, 0, 4),
                None,
                None,
                _Action(B, 0, 1),
                _Action(B, 0, 0),
            ],
            1: [
                None,
                None,
                _Action(F, 0, 2),
                _Action(F, 0, 3),
                None,
                None,
                _Action(B, 0, 3),
                _Action(B, 0, 2),
                None,
                None,
            ],
        }


class ScheduleWithW(PipelineScheduleMulti):
    n_stages = 4
    rank_stages = {
        0: [0, 2],
        1: [1, 3],
    }

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
        )

        # Needs to be updated as part of all schedules using "W"
        self.use_full_backward = False

        # Go through two microbatches
        # F0_0 F1_0 F0_2 F1_2 None B0_2 W0_2 B0_0 B1_2 W0_0 B1_0 W1_2 W1_0
        # None F0_1 F1_1 F0_3 B0_3 F1_3 B0_1 B1_3 W0_3 B1_1 W0_1 W1_3 W1_1
        self.pipeline_order = {
            0: [
                _Action(F, 0, 0),
                _Action(F, 1, 0),
                _Action(F, 0, 2),
                _Action(F, 1, 2),
                None,
                _Action(B, 0, 2),
                _Action(W, 0, 2),
                _Action(B, 0, 0),
                _Action(B, 1, 2),
                _Action(W, 0, 0),
                _Action(B, 1, 0),
                _Action(W, 1, 2),
                _Action(W, 1, 0),
            ],
            1: [
                None,
                _Action(F, 0, 1),
                _Action(F, 1, 1),
                _Action(F, 0, 3),
                _Action(B, 0, 3),
                _Action(F, 1, 3),
                _Action(B, 0, 1),
                _Action(B, 1, 3),
                _Action(W, 0, 3),
                _Action(B, 1, 1),
                _Action(W, 0, 1),
                _Action(W, 1, 3),
                _Action(W, 1, 1),
            ],
        }
