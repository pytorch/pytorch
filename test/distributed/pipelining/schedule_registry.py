# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# This file is a Schedule zoo for testing torch.distributed.pipelining.
# It includes schedules designed purely for testing purposes
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.distributed.pipelining.schedules import _ComputationType, _Action, PipelineScheduleMulti
from torch.distributed.pipelining.stage import _PipelineStageBase

F = _ComputationType.FORWARD
B = _ComputationType.BACKWARD

class ScheduleUnevenLooped(PipelineScheduleMulti):
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
            0: [_Action(F, 0, 0), None, None, _Action(F, 0, 3), None, None, _Action(B, 0, 3), None, _Action(B, 0, 0)],
            1: [None, _Action(F, 0, 1), _Action(F, 0, 2), None, None, _Action(B, 0, 2), _Action(B, 0, 1), None],
        }
