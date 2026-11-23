# Owner(s): ["module: inductor"]

from types import SimpleNamespace

import sympy

import torch
from torch._inductor import config, ir
from torch._inductor.codegen.memory_planning import (
    Allocation,
    AllocationPool,
    LiveRange,
    MemoryPlanner,
    TemporalSplit,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import align
from torch.utils._sympy.functions import Max


class MemoryPlanningPoolsTest(TestCase):
    class FakeBuffer:
        def __init__(self, name):
            self._name = name

        def get_device(self):
            return torch.device("cpu")

        def get_layout(self):
            return SimpleNamespace(size=(), stride=())

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


class PoolAlignmentTest(TestCase):
    @staticmethod
    def make_allocation(name, nbytes, live_range, size_hint=64):
        node = ir.Buffer(
            name=name,
            layout=ir.FixedLayout(torch.device("cpu"), torch.uint8, [nbytes], [1]),
        )
        return Allocation(node, live_range, size_hint=size_hint, symbolic_size=nbytes)

    def test_symbolic_pool_alignment(self):
        a, b, c, d = sympy.symbols("a b c d", integer=True)
        first = self.make_allocation("first", a, LiveRange(0, 1))
        second = self.make_allocation("second", b, LiveRange(1, 2))
        tail = self.make_allocation("tail", c, LiveRange(0, 3))
        anchor = self.make_allocation("anchor", sympy.Integer(1), LiveRange(0, 3))
        first.mark_allocated()
        second.mark_allocated()
        pool = AllocationPool(torch.device("cpu"), TemporalSplit([first, second]))
        pool.allocate_at_end(tail)
        pool.allocate_at_end(anchor)
        prefix = Max(align(a), align(b))
        self.assertEqual(pool.root.get_symbolic_size(), prefix + align(c) + 1)

        replacement = self.make_allocation("replacement", d, LiveRange(2, 3))
        self.assertTrue(pool.allocate(replacement, is_last=False))
        prefix = Max(align(a), align(b), align(d))
        self.assertEqual(pool.root.get_symbolic_size(), prefix + align(c) + 1)
        pool.finalize("pool0")
        self.assertEqual(
            [block.offset for block in (first, second, tail, anchor, replacement)],
            [0, 0, prefix, prefix + align(c), 0],
        )


class UnbackedPoolTest(TestCase):
    @staticmethod
    def plan(*blocks):
        planner = MemoryPlanner(wrapper=None)
        planner.buffer_groups = [
            MemoryPlanningPoolsTest.FakeBufferGroup(block, is_output=False)
            for block in blocks
        ]
        with config.patch(memory_pool="intermediates"):
            planner.allocate_groups()

    def test_unbacked_symbol_available_from_first_use(self):
        make = PoolAlignmentTest.make_allocation
        u0, u1 = sympy.symbols("u0 u1", integer=True, positive=True)
        static = make("static", sympy.Integer(64), LiveRange(1, 2))
        first = make("first", u0, LiveRange(2, 3))
        second = make("second", u0, LiveRange(4, 5))
        late = make("late", u1, LiveRange(6, 7))
        self.plan(static, first, second, late)

        self.assertEqual(
            [block.earliest_available for block in (static, first, second, late)],
            [None, 2, 2, 6],
        )
        self.assertIs(first.pool, second.pool)
        self.assertEqual(len({id(block.pool) for block in (static, first, late)}), 3)

    def test_pool_begins_after_symbols_of_every_member(self):
        make = PoolAlignmentTest.make_allocation
        u0, u1 = sympy.symbols("u0 u1", integer=True, positive=True)
        # u0 is first used at timestep 5 and u1 at 7, so once a and b share a pool,
        # c must not move the start of that pool before 7.
        a = make("a", u0, LiveRange(9, 10), size_hint=256)
        b = make("b", u1, LiveRange(12, 13), size_hint=256)
        c = make("c", sympy.Integer(128), LiveRange(6, 7), size_hint=128)
        d = make("d", u0, LiveRange(5, 6))
        e = make("e", u1, LiveRange(7, 8))
        self.plan(a, b, c, d, e)

        self.assertIs(a.pool, b.pool)
        self.assertIsNot(c.pool, a.pool)
        for block in (a, b, c, d, e):
            pool_begin = block.pool.root.get_live_ranges().begin
            if block.earliest_available is not None:
                self.assertLessEqual(block.earliest_available, pool_begin)


if __name__ == "__main__":
    run_tests()
