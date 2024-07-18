# Owner(s): ["module: inductor"]
import contextlib

import torch
from torch._inductor.dependencies import MemoryDep

from torch._inductor.graph import GraphLowering
from torch._inductor.ir import Buffer, FixedLayout, Pointwise
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import sympy_index_symbol
from torch._inductor.virtualized import ops, V

from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


class TestDependencies(InductorTestCase):
    def _create_buffer(self, name, shape, dtype=torch.float32):
        return Buffer(name, FixedLayout(torch.device(GPU_TYPE), dtype, shape))

    def setUp(self):
        super().setUp()

        class DummyModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        self._gm = torch.fx.symbolic_trace(DummyModule())
        self._graph = GraphLowering(self._gm)

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(V.set_graph_handler(self._graph))

    def tearDown(self):
        self._stack.close()
        super().tearDown()

    def test_bucketize_dependencies(self):
        offsets = self._create_buffer("offsets", (1025,), torch.int32)

        def inner_fn(index):
            idx = index[0]
            return ops.bucketize(
                values=idx,
                offsets_name=offsets.get_name(),
                offsets_size=offsets.get_size()[0],
                indexing_dtype=torch.int32,
                right=True,
            )

        pointwise = Pointwise.create(
            device=torch.device(GPU_TYPE),
            dtype=torch.int32,
            inner_fn=inner_fn,
            ranges=[1024 * 4],
        )

        self.assertEqual(len(pointwise.get_reads()), 1)

    def test_get_offset(self):
        x = sympy_index_symbol("x")
        y = sympy_index_symbol("y")
        var_ranges = {
            x: 1024,
            y: 2048,
        }
        dep1 = MemoryDep(
            "dep1",
            x * 2048 + y,
            list(var_ranges.keys()),
            list(var_ranges.values()),
        )
        dep2 = MemoryDep(
            "dep2",
            x * 2048 + y + 1024,
            list(var_ranges.keys()),
            list(var_ranges.values()),
        )
        self.assertEqual(dep1.get_offset(), 0)
        self.assertEqual(dep2.get_offset(), 1024)

    def test_normalize_with_stride_order_equal(self):
        x = sympy_index_symbol("x")
        y = sympy_index_symbol("y")
        var_ranges = {
            x: 1024,
            y: 2048,
        }

        loop_order1 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y,
            [x, y],
            [1024, 2048],
        )
        loop_order2 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y,
            [y, x],
            [2048, 1024],
        )
        self.assertTrue(loop_order1 != loop_order2)
        normalized_loop_order1 = loop_order1.normalize_with_stride_order()
        normalized_loop_order2 = loop_order2.normalize_with_stride_order()
        self.assertTrue(normalized_loop_order1 == normalized_loop_order2)

    def test_normalize_with_stride_order_unequal(self):
        x = sympy_index_symbol("x")
        y = sympy_index_symbol("y")
        var_ranges = {
            x: 1024,
            y: 2048,
        }

        loop_order1 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y,
            [x, y],
            [1024, 2048],
        )
        loop_order2 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y + 5,
            [y, x],
            [2048, 1024],
        )
        self.assertTrue(loop_order1 != loop_order2)
        normalized_loop_order1 = loop_order1.normalize_with_stride_order()
        normalized_loop_order2 = loop_order2.normalize_with_stride_order()
        # unequal due to different offset
        self.assertTrue(normalized_loop_order1 != normalized_loop_order2)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU and HAS_GPU:
        run_tests("sympy")
