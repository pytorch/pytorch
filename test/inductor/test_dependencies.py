# Owner(s): ["module: inductor"]
import contextlib
import itertools
from unittest.mock import patch

import sympy

import torch
from torch._inductor.codegen.cpp_utils import CppCSEVariable
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    Buffer,
    ExternKernel,
    FixedLayout,
    Pointwise,
    ShapeAsConstantBuffer,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import sympy_index_symbol
from torch._inductor.virtualized import ops, V
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.symbol import make_symbol, SymT
from torch.utils._sympy.value_ranges import ValueRanges


class TestDependencies(InductorTestCase):
    def _create_buffer(self, name, shape, dtype=torch.float32):
        return Buffer(
            name=name,
            layout=FixedLayout(torch.device(GPU_TYPE), dtype=dtype, size=shape),
        )

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

    def test_bucketize_dependencies_no_sorter(self):
        offsets = self._create_buffer("offsets", (1025,), torch.int32)

        def inner_fn(index):
            idx = index[0]
            return ops.bucketize(
                values=idx,
                boundaries=(
                    offsets.get_name(),
                    offsets.get_size()[-1],
                    offsets.get_size()[0] * offsets.get_stride()[0],
                    offsets.get_stride()[-1],
                ),
                boundary_indices=0,
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

    def test_bucketize_dependencies_sorter(self):
        offsets = self._create_buffer("offsets", (1025,), torch.int32)
        sorter = self._create_buffer("sorter", (1025,), torch.int32)

        def inner_fn(index):
            idx = index[0]
            return ops.bucketize(
                values=idx,
                boundaries=(
                    offsets.get_name(),
                    offsets.get_size()[-1],
                    offsets.get_size()[0] * offsets.get_stride()[0],
                    offsets.get_stride()[-1],
                ),
                boundary_indices=0,
                indexing_dtype=torch.int32,
                right=True,
                sorter=(
                    sorter.get_name(),
                    sorter.get_stride()[-1],
                ),
                sorter_indices=0,
            )

        pointwise = Pointwise.create(
            device=torch.device(GPU_TYPE),
            dtype=torch.int32,
            inner_fn=inner_fn,
            ranges=[1024 * 4],
        )

        self.assertEqual(len(pointwise.get_reads()), 2)

    def test_extern_kernel_nested_ir_args_are_dependencies(self):
        data = self._create_buffer("data", (4,))
        index = self._create_buffer("index", (4,), torch.int64)
        value = self._create_buffer("value", (4,))
        out = self._create_buffer("out", (4,))
        shape_arg = ShapeAsConstantBuffer(expr=sympy_index_symbol("nested_size"))

        extern = ExternKernel(
            name="extern",
            layout=out.layout,
            inputs=[data],
            constant_args=([None, index, shape_arg],),
            kwargs={"value": {"nested": (value, shape_arg)}},
        )

        reads = {dep.name for dep in extern.get_read_writes().reads}
        self.assertEqual(reads, {"data", "index", "value"})

    def test_non_overlapping_addresses(self):
        # Check accepted maps independently, by enumerating their addresses.
        for rank in (1, 2, 3):
            variables = tuple(sympy_index_symbol(f"d{i}") for i in range(rank))
            accepted = 0
            for sizes in itertools.product((2, 3), repeat=rank):
                for strides in itertools.product((1, 2, 3, 6), repeat=rank):
                    index = 7 + sum(s * v for s, v in zip(strides, variables))
                    dep = MemoryDep("buf", index, variables, sizes)
                    if dep.is_non_overlapping():
                        addresses = [
                            7 + sum(s * x for s, x in zip(strides, point))
                            for point in itertools.product(*(range(n) for n in sizes))
                        ]
                        self.assertEqual(len(set(addresses)), len(addresses))
                        accepted += 1
            self.assertGreater(accepted, 0)

    def test_non_overlapping_pitched_and_symbolic(self):
        row, col = sympy_index_symbol("row"), sympy_index_symbol("col")
        batch = sympy.Symbol("batch", integer=True, positive=True)
        for pitch, width in ((1088, 1056), (24, 20)):
            for rows in (7, batch):
                for variables, sizes in (
                    ((row, col), (rows, width)),
                    ((col, row), (width, rows)),
                ):
                    dep = MemoryDep("buf", pitch * row + col + 13, variables, sizes)
                    self.assertTrue(dep.is_non_overlapping())
        unknown_pitch = sympy.Symbol("pitch", integer=True, positive=True)
        dep = MemoryDep("buf", unknown_pitch * row + col, (row, col), (3, 12))
        # A favorable hint cannot substitute for a proven stride bound.
        with patch.object(
            self._graph.sizevars,
            "optimization_hint",
            side_effect=lambda expr, fallback=None: 4096
            if expr == unknown_pitch
            else int(expr),
        ):
            self.assertFalse(dep.is_non_overlapping())

    def test_non_overlapping_rejects_unproved_maps(self):
        row, col = sympy_index_symbol("row"), sympy_index_symbol("col")
        for index in (
            8 * row + col,  # overlapping rows
            col,  # omitted axis
            16 * row - col,
            16 * row + ModularIndexing(col, 1, 4),
            16 * row + FloorDiv(col, 2),
            row * col,
            16 * row + col**2,
            16 * row + col + make_symbol(SymT.TMP, 0),  # per-iteration offset
        ):
            with self.subTest(index=index):
                self.assertFalse(
                    MemoryDep("buf", index, (row, col), (3, 12)).is_non_overlapping()
                )

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

    def test_cpp_cse_value_expr_tracks_dependent_itervars(self):
        x = sympy_index_symbol("x")

        class DummyCSE:
            varname_map = {}

        class DummyKernel:
            cse = DummyCSE()
            itervars = {x}

        var = CppCSEVariable("tmp0", ValueRanges.unknown(), torch.int64)
        with V.set_kernel_handler(DummyKernel()):
            var.update_on_args("value_expr", (x + 1, torch.int64), {})

        self.assertTrue(var.depends_on(x))

    def test_normalize_with_stride_order_equal(self):
        x = sympy_index_symbol("x")
        y = sympy_index_symbol("y")

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

    def test_normalize_with_ranges(self):
        d0 = sympy_index_symbol("d0")
        d1 = sympy_index_symbol("d1")
        x = sympy_index_symbol("x")
        r = sympy_index_symbol("r")
        dep = MemoryDep("buf", 100 * d0 + 2 * d1 + 1, (d0, d1), (4, 8))

        normalized = dep.normalize_with_ranges((x, r), (2, 16))
        self.assertIsNotNone(normalized)
        for x_value in range(2):
            for r_value in range(16):
                flat = 16 * x_value + r_value
                expected = 100 * (flat // 8) + 2 * (flat % 8) + 1
                self.assertEqual(
                    normalized.index.subs({x: x_value, r: r_value}), expected
                )

        broadcast = MemoryDep("buf", d0, (d0,), (64,))
        self.assertIsNone(broadcast.normalize_with_ranges((x, r), (64, 64)))

        indirect = MemoryDep("buf", sympy_index_symbol("tmp0"), (d0,), (64,))
        self.assertIsNone(indirect.normalize_with_ranges((x, r), (8, 8)))

        aligned = MemoryDep("buf", 128 * d0 + 2 * d1 + 1, (d0, d1), (4, 128))
        normalized_aligned = aligned.normalize_with_ranges((x, r), (64, 8))
        self.assertIsNotNone(normalized_aligned)
        self.assertEqual(
            normalized_aligned.index,
            128 * FloorDiv(x, 16) + 16 * ModularIndexing(x, 1, 16) + 2 * r + 1,
        )

        scalar = MemoryDep("buf", d0 - d0 + 5, (), ())
        normalized_scalar = scalar.normalize_with_ranges((x, r), (64, 64))
        self.assertIsNotNone(normalized_scalar)
        self.assertEqual(normalized_scalar.index, 5)

    def test_normalize_with_stride_order_unequal(self):
        x = sympy_index_symbol("x")
        y = sympy_index_symbol("y")

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
