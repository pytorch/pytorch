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
    _add_send_recv,
    _add_unshard_reshard,
    _format_pipeline_order,
    _validate_pipeline_order,
    B,
    F,
    RECV_F,
    RESHARD,
    SEND_B,
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
    def setUp(self):
        # Define a list of test cases with varying num_local_stages, num_microbatches, and group_size
        # These should succeed since num_microbatches % group_size == 0
        self.test_cases = [
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

    @parametrize(
        "ScheduleClass",
        [ScheduleInterleaved1F1B, ScheduleLoopedBFS],
    )
    def test_pipeline_order(self, ScheduleClass):
        for num_local_stages, num_microbatches, group_size in self.test_cases:
            with self.subTest(
                num_local_stages=num_local_stages,
                num_microbatches=num_microbatches,
                group_size=group_size,
            ):
                if num_microbatches % group_size != 0:
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

    @parametrize(
        "ScheduleClass",
        [ScheduleFlexibleInterleaved1F1B],
    )
    def test_pipeline_order_flex_and_zero_bubble(self, ScheduleClass):
        for num_local_stages, num_microbatches, group_size in self.test_cases:
            with self.subTest(
                num_local_stages=num_local_stages,
                num_microbatches=num_microbatches,
                group_size=group_size,
            ):
                warmups_ops_last_stage = (num_local_stages - 1) * (
                    num_microbatches // max(1, num_microbatches // group_size)
                )
                warmup_ops = warmups_ops_last_stage + 2 * (group_size - 1)
                warmup_ops = min(warmup_ops, num_microbatches * num_local_stages)

                for i in range(2):
                    num_stages = num_local_stages * group_size
                    stages = [
                        MockPipelineStage(group_size=group_size, num_stages=num_stages)
                        for i in range(num_local_stages)
                    ]
                    schedule = ScheduleClass(
                        stages, num_microbatches, enable_zero_bubble=(i == 0)
                    )
                    formatted_pipeline_order = _format_pipeline_order(
                        schedule.pipeline_order
                    )
                    # print(formatted_pipeline_order)
                    _validate_pipeline_order(
                        schedule.pipeline_order,
                        num_microbatches,
                        num_stages,
                        enable_zero_bubble=(i == 0),
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
            ("1UNSHARD", _Action(1, UNSHARD, None)),
            ("3RESHARD", _Action(3, RESHARD, None)),
            ("2SEND_B2", _Action(2, SEND_B, 2)),
            ("1RECV_F1", _Action(1, RECV_F, 1)),
        ],
    )
    def test_action_parse(self, action_str_and_ref):
        """Test that actions can be parsed from strings and round-tripped back to the same strings."""
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
        """Test the lowering pass that takes a 'compute only' schedule (with only F,B,W ops) and adds
        FSDP unshard/reshard operations to the schedule.  This is just part of the process of adding communication
        ops and producing a complete schedule.
        """
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

    @parametrize(
        "test_info",
        [
            {
                "compute": {
                    0: ["0F0", "0F1", "   ", "0B0", "   ", "0B1"],
                    1: ["   ", "1F0", "1B0", "1F1", "1B1", "   "],
                },
                "comms": {
                    0: [
                        "0F0",
                        "0SEND_F0",
                        "0F1",
                        "0SEND_F1",
                        "0RECV_B0",
                        "0B0",
                        "0RECV_B1",
                        "0B1",
                    ],
                    1: [
                        "1RECV_F0",
                        "1RECV_F1",
                        "1F0",
                        "1B0",
                        "1SEND_B0",
                        "1F1",
                        "1B1",
                        "1SEND_B1",
                    ],
                },
                "stage_to_rank": lambda stage_idx: stage_idx,
                "num_stages": 2,
            },
        ],
    )
    def test_send_recv(self, test_info):
        """Tests the lowering pass that adds send/recv ops to a compute-only schedule."""
        compute_sch = {
            rank: self._parse_actions(test_info["compute"][rank])
            for rank in test_info["compute"]
        }
        expected_comms_sch = {
            rank: self._parse_actions(test_info["comms"][rank])
            for rank in test_info["comms"]
        }

        comms_sch = _add_send_recv(
            compute_sch, test_info["stage_to_rank"], test_info["num_stages"]
        )
        for rank in expected_comms_sch:
            for i, (expected, actual) in enumerate(
                zip(expected_comms_sch[rank], comms_sch[rank])
            ):
                self.assertEqual(
                    expected,
                    actual,
                    (
                        f"Mismatch on rank {rank} at position {i}."
                        f"\nExpected: {expected_comms_sch[rank]}"
                        f"\nActual:   {comms_sch[rank]}"
                    ),
                )
            self.assertEqual(len(comms_sch[rank]), len(expected_comms_sch[rank]))


instantiate_parametrized_tests(TestScheduleLowering)

if __name__ == "__main__":
    run_tests()
