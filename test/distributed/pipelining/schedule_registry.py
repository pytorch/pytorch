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
        # Note(whc) - it might be easier to work with thes schedules by writing them as a list of
        # ["0F0", ...] and then parsing them in the test infra to turn them into actions.
        self.pipeline_order = {
            0: [
                _Action(0, F, 0),
                None,
                None,
                _Action(3, F, 0),
                _Action(3, B, 0),
                None,
                None,
                _Action(0, B, 0),
            ],
            1: [
                None,
                _Action(1, F, 0),
                _Action(2, F, 0),
                None,
                None,
                _Action(2, B, 0),
                _Action(1, B, 0),
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
                _Action(0, F, 0),
                _Action(1, F, 0),
                None,
                None,
                _Action(4, F, 0),
                _Action(4, B, 0),
                None,
                None,
                _Action(1, B, 0),
                _Action(0, B, 0),
            ],
            1: [
                None,
                None,
                _Action(2, F, 0),
                _Action(3, F, 0),
                None,
                None,
                _Action(3, B, 0),
                _Action(2, B, 0),
                None,
                None,
            ],
        }


class ScheduleWithW(PipelineScheduleMulti):
    n_stages = 4
    num_microbatches = 2
    rank_stages = {
        0: [0, 2],
        1: [1, 3],
    }

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        enable_zero_bubble: bool = True,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
        )

        # Needs to be updated as part of all schedules using "W"
        self.use_full_backward = False

        # Go through two microbatches
        self.pipeline_order = {
            0: [
                _Action(0, F, 0),
                _Action(0, F, 1),
                _Action(2, F, 0),
                _Action(2, F, 1),
                None,
                _Action(2, B, 0),
                _Action(2, W, 0),
                _Action(0, B, 0),
                _Action(2, B, 1),
                _Action(0, W, 0),
                _Action(0, B, 1),
                _Action(2, W, 1),
                _Action(0, W, 1),
            ],
            1: [
                None,
                _Action(1, F, 0),
                _Action(1, F, 1),
                _Action(3, F, 0),
                _Action(3, B, 0),
                _Action(3, F, 1),
                _Action(1, B, 0),
                _Action(3, B, 1),
                _Action(3, W, 0),
                _Action(1, B, 1),
                _Action(1, W, 0),
                _Action(3, W, 1),
                _Action(1, W, 1),
            ],
        }
