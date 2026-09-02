# Owner(s): ["module: inductor"]
import contextlib
from unittest.mock import Mock

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
from torch._inductor.loop_body import _MaskStoresHandler, LoopBody
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import sympy_index_symbol
from torch._inductor.virtualized import ops, V
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
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

    def test_masked_store_records_full_write(self):
        """
        A masked store is deliberately recorded as a full write over the
        expanded domain. That over-approximation is what keeps WAW/WAR ordering
        edges intact; the resulting imprecision is handled by refusing in-place
        reuse (see SchedulerNode.can_inplace).
        """
        from torch._inductor.dependencies import extract_read_writes

        def fn(index):
            (x,) = index
            mask = ops.lt(ops.index_expr(x, torch.int32), ops.constant(48, torch.int32))
            ops.masked_store("out", x, ops.constant(1.0, torch.float32), mask)

        rw = extract_read_writes(fn, [64])
        writes = [dep for dep in rw.writes if isinstance(dep, MemoryDep)]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0].name, "out")
        # Recorded over the whole 64-element range even though the mask may
        # exclude a tail, and with no store mode (no atomic masked store).
        self.assertEqual(writes[0].get_numel(), 64)
        self.assertEqual(writes[0].mode, None)

    def test_masked_expansion_rewrites_stores_and_keeps_raw_loads(self):
        from torch._inductor.loop_body import MASKED_EXPANSION_BANNED_OPS
        from torch.utils._sympy.functions import ModularIndexing

        x = sympy_index_symbol("x")

        def fn(index, rindex):
            (i,) = index
            edge = ops.lt(ops.index_expr(i, torch.int64), ops.constant(2, torch.int64))
            other = ops.masked(edge, lambda: ops.load("other", i), 0.0)
            ops.store("out", i, ops.add(ops.load("inp", i), other))

        body = LoopBody(fn, ([x], []), {x: 4}, [x], [])
        expanded = body.expand_dimension_for_pointwise_node_with_masked_stores(0, 6)

        self.assertEqual(expanded.sizes, ((6,), ()))
        self.assertTrue(expanded.has_op("masked_store"))
        self.assertFalse(expanded.has_op("store"))
        self.assertFalse(
            any(e.has(ModularIndexing) for e in expanded.indexing_exprs.values())
        )
        # The tail predicate is combined into the nested mask.
        masked_calls = expanded.root_block.graph.find_nodes(op="call_module")
        self.assertTrue(
            any(str(n.target).startswith("masked_subblock") for n in masked_calls)
        )
        self.assertEqual(
            len(
                expanded.root_block.graph.find_nodes(
                    op="call_method", target="logical_and"
                )
            ),
            1,
        )
        # Expanding to the current range is a no-op.
        self.assertIs(
            body.expand_dimension_for_pointwise_node_with_masked_stores(0, 4), body
        )
        # Bodies with banned ops are rejected before any mutation.
        self.assertIn("indirect_indexing", MASKED_EXPANSION_BANNED_OPS)

        def indirect(index, rindex):
            (i,) = index
            j = ops.indirect_indexing(ops.load("idx", i), 4)
            ops.store("out", i, ops.load("inp", j))

        bad = LoopBody(indirect, ([x], []), {x: 4}, [x], [])
        with self.assertRaisesRegex(AssertionError, "indirect_indexing"):
            bad.expand_dimension_for_pointwise_node_with_masked_stores(0, 6)
        with self.assertRaisesRegex(AssertionError, "masked_store"):
            expanded.expand_dimension_for_pointwise_node_with_masked_stores(0, 8)

    def test_masked_store_disables_inplace_reuse(self):
        body = object.__new__(LoopBody)
        body.op_counts = {"masked_store": 1}
        node = object.__new__(SchedulerNode)
        node._body = body
        node.node = None
        node.outputs = []

        self.assertFalse(node.can_inplace(object()))

    def test_masked_expansion_combines_nested_mask(self):
        inner = Mock()
        inner.logical_and.return_value = "combined"
        body = Mock()

        _MaskStoresHandler(inner, "outer").masked("inner", body, "other")

        inner.logical_and.assert_called_once_with("outer", "inner")
        inner.masked.assert_called_once_with("combined", body, "other")

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
