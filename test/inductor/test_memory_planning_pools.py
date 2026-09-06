# Owner(s): ["module: inductor"]

from types import SimpleNamespace

import torch
from torch._inductor import config
from torch._inductor.codegen.memory_planning import Allocation, LiveRange, MemoryPlanner
from torch._inductor.test_case import run_tests, TestCase


class MemoryPlanningPoolsTest(TestCase):
    class FakeBuffer:
        def __init__(self, name):
            self._name = name

        def get_device(self):
            return torch.device("cpu")

        def get_layout(self):
            return SimpleNamespace(size=())

        def get_name(self):
            return self._name

    class FakeBufferGroup:
        def __init__(self, allocation, is_output):
            self.allocation = allocation
            self.is_output = is_output

        def make_allocation(self):
            pass

    def make_group(self, name, size, is_output):
        allocation = Allocation(
            self.FakeBuffer(name),
            LiveRange(0, 10),
            size_hint=size,
            symbolic_size=size,
        )
        return self.FakeBufferGroup(allocation, is_output)

    def test_memory_pool_modes(self):
        expected_patterns = (
            ("none", (0, 1, 2, 3)),
            ("intermediates", (0, 1, 2, 2)),
            ("outputs", (0, 0, 1, 1)),
            ("combined", (0, 0, 0, 0)),
        )

        for memory_pool, expected_pool_pattern in expected_patterns:
            with self.subTest(memory_pool=memory_pool):
                groups = [
                    self.make_group("output_0", 16, True),
                    self.make_group("output_1", 32, True),
                    self.make_group("intermediate_0", 64, False),
                    self.make_group("intermediate_1", 48, False),
                ]
                planner = MemoryPlanner(wrapper=None)
                planner.buffer_groups = groups

                with config.patch(memory_pool=memory_pool):
                    planner.allocate_groups()

                pool_indices = {}
                actual_pool_pattern = tuple(
                    pool_indices.setdefault(
                        id(group.allocation.pool), len(pool_indices)
                    )
                    for group in groups
                )
                self.assertEqual(expected_pool_pattern, actual_pool_pattern)


if __name__ == "__main__":
    run_tests()
