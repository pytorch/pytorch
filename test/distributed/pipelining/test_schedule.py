# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import logging
from typing import List

import torch
from torch.distributed.pipelining import (
    ScheduleFlexibleInterleaved1F1B,
    ScheduleInterleaved1F1B,
    ScheduleLoopedBFS,
)
from torch.distributed.pipelining.schedules import (
    _Action,
    _add_unshard_reshard,
    _format_pipeline_order,
    _validate_pipeline_order,
    B,
    F,
    RESHARD,
    UNSHARD,
    W,
)
from torch.distributed.pipelining.stage import _PipelineStageBase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

logger = logging.getLogger(__name__)
torch.manual_seed(0)


class MockPipelineStage(_PipelineStageBase):
    def __init__(self, *args, **kwargs):
        # Mock the necessary attributes
        self.num_stages = kwargs.get("num_stages", 1)
        self.group_size = kwargs.get("group_size", 1)
        self.group_rank = kwargs.get("group_rank", 0)
        self.group = kwargs.get("group", None)
        self.stage_index_to_group_rank = kwargs.get("stage_index_to_group_rank", None)

    def _create_grad_recv_info(self, *args, **kwargs):
        return None

    def _prepare_forward_infra(self, n_microbatches):
        pass

    def _prepare_backward_infra(self, n_microbatches):
        pass


class TestSchedulePlan(TestCase):
    @parametrize(
        "ScheduleClass",
        [ScheduleFlexibleInterleaved1F1B, ScheduleInterleaved1F1B, ScheduleLoopedBFS],
    )
    def test_pipeline_order(self, ScheduleClass):
        # Define a list of test cases with varying num_local_stages, num_microbatches, and group_size
        # These should succeed since num_microbatches % group_size == 0
        test_cases = [
            # small number of stages
            (2, 2, 2),
            (2, 4, 4),
            (2, 8, 2),
            (2, 8, 4),
            (2, 8, 8),
            (4, 4, 4),
            (4, 8, 4),
            (4, 8, 8),
            # large microbatches
            (4, 16, 4),
            (4, 32, 4),
            (4, 64, 4),
            # large groups
            (4, 16, 16),
            (4, 32, 32),
            (4, 128, 64),
            # odd num pipeline stages
            (3, 2, 2),
            (3, 8, 2),
            (3, 12, 4),
            # odd group_sizes
            (4, 6, 3),
            (4, 10, 5),
            # n_mb non divisible by group_size
            (2, 3, 4),
            (2, 4, 4),
            (2, 10, 4),
            (2, 15, 4),
        ]
        for num_local_stages, num_microbatches, group_size in test_cases:
            with self.subTest(
                num_local_stages=num_local_stages,
                num_microbatches=num_microbatches,
                group_size=group_size,
            ):
                only_run_in_flex_pp = num_microbatches % group_size != 0
                if only_run_in_flex_pp and not isinstance(
                    ScheduleClass, ScheduleFlexibleInterleaved1F1B
                ):
                    continue

                print(f"{num_local_stages=} {num_microbatches=} {group_size=}")
                num_stages = num_local_stages * group_size
                stages = [
                    MockPipelineStage(group_size=group_size, num_stages=num_stages)
                    for i in range(num_local_stages)
                ]

                schedule = ScheduleClass(stages, num_microbatches)
                formatted_pipeline_order = _format_pipeline_order(
                    schedule.pipeline_order
                )
                # print(formatted_pipeline_order)
                _validate_pipeline_order(
                    schedule.pipeline_order, num_microbatches, num_stages
                )


instantiate_parametrized_tests(TestSchedulePlan)


class TestScheduleLowering(TestCase):
    """Tests lowering passes that convert simple compute-only (FBW) schedules into compute+comms schedules"""

    def _parse_actions(self, actions: List[str]) -> List[_Action]:
        return [_Action.from_str(s) for s in actions]

    @parametrize(
        "action_str_and_ref",
        [
            ("1F0", _Action(1, F, 0)),
            ("2B1", _Action(2, B, 1)),
            ("0W3", _Action(0, W, 3)),
            ("1UNSHARD", _Action(1, UNSHARD)),
            ("3RESHARD", _Action(3, RESHARD)),
        ],
    )
    def test_action_parse(self, action_str_and_ref):
        act_str, ref = action_str_and_ref
        act = _Action.from_str(act_str)
        self.assertEqual(act, ref)
        self.assertEqual(act_str, act.__repr__())

    @parametrize(
        "test_info",
        [
            {
                "compute": ["0F0", "0F1", "   ", "0B0", "0B1"],
                "comms": ["0UNSHARD", "0F0", "0F1", "0B0", "0B1", "0RESHARD"],
            },
        ],
    )
    def test_unshard_reshard(self, test_info):
        compute_sch = self._parse_actions(test_info["compute"])
        expected_comms_sch = self._parse_actions(test_info["comms"])

        comms_sch = _add_unshard_reshard(compute_sch)
        for expected, actual in zip(expected_comms_sch, comms_sch):
            self.assertEqual(
                expected,
                actual,
                (
                    f"Mismatch: expected action {expected} but found {actual}."
                    f"\nWhole Schedule: {comms_sch}"
                ),
            )


instantiate_parametrized_tests(TestScheduleLowering)

if __name__ == "__main__":
    run_tests()
