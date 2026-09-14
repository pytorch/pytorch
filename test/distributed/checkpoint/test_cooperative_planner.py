# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed.checkpoint.cooperative_planner import (
    _create_cooperative_plans,
    _read_signature,
)
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    ReadItem,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def _read_item(fqn: str) -> ReadItem:
    index = MetadataIndex(fqn, (0,))
    return ReadItem(
        type=LoadItemType.TENSOR,
        dest_index=index,
        dest_offsets=torch.Size([0]),
        storage_index=index,
        storage_offsets=torch.Size([0]),
        lengths=torch.Size([8]),
    )


def _plan(*items: ReadItem, shape=(8,)) -> LoadPlan:
    metadata = {
        _read_signature(item): (shape, "torch.float32", "cuda", 32)
        for item in items
    }
    return LoadPlan(items=list(items), planner_data=metadata)


class TestCooperativeLoadPlanner(TestCase):
    def test_deduplicates_and_byte_balances_readers(self):
        weight = _read_item("model.weight")
        bias = _read_item("model.bias")
        plans = _create_cooperative_plans(
            [_plan(weight, bias), _plan(bias, weight)],
            global_ranks=(0, 1),
            min_tensor_bytes=0,
        )

        self.assertEqual(2, sum(len(plan.items) for plan in plans))
        schedules = [plan.planner_data.schedule for plan in plans]
        self.assertEqual(schedules[0], schedules[1])
        self.assertEqual({0, 1}, {task.src for task in schedules[0]})
        self.assertEqual(
            ["model.bias", "model.weight"],
            [task.read_item.dest_index.fqn for task in schedules[0]],
        )

    def test_keeps_native_reads_on_metadata_mismatch(self):
        weight = _read_item("model.weight")
        plans = [_plan(weight, shape=(8,)), _plan(weight, shape=(4,))]
        result = _create_cooperative_plans(
            plans,
            global_ranks=(0, 1),
            min_tensor_bytes=0,
        )

        self.assertEqual([1, 1], [len(plan.items) for plan in result])
        self.assertEqual([(), ()], [plan.planner_data.schedule for plan in result])


if __name__ == "__main__":
    run_tests()
