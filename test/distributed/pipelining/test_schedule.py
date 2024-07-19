# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import logging

import torch
from torch.distributed.pipelining import (
    ScheduleFlexibleInterleaved1F1B,
    ScheduleInterleaved1F1B,
    ScheduleLoopedBFS,
)
from torch.distributed.pipelining.schedules import (
    _format_pipeline_order,
    _validate_pipeline_order,
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


if __name__ == "__main__":
    run_tests()
