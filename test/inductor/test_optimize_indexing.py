# Owner(s): ["module: inductor"]

import operator

import sympy

import torch
from torch._inductor import config
from torch._inductor.codegen.common import deduce_output_dtype_by_name
from torch._inductor.loop_body import LoopBody
from torch._inductor.optimize_indexing import (
    convert_index_expr_to_value_expr,
    remove_redundant_argreduce_indices,
)
from torch._inductor.virtualized import V
from torch.fx import Graph
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._sympy.value_ranges import ValueRanges


class _FakeSizeVars:
    @staticmethod
    def simplify_with_ranges(expr, var_ranges):
        return expr

    @staticmethod
    def statically_known_equals(left, right):
        return sympy.expand(left - right) == 0


class _FakeGraph:
    sizevars = _FakeSizeVars()


class TestOptimizeIndexing(TestCase):
    @staticmethod
    def _make_loop_body(
        graph,
        bounds,
        indexing_exprs,
        replacement_vals,
        subgraphs=(),
        indirect_vars=(),
    ):
        class FakeBounds:
            def __init__(self):
                self.replacement_vals = replacement_vals

            def get_bounds(self):
                return bounds

        class FakeBlock:
            def __init__(self, graph):
                self.graph = graph

        class FakeLoopBody:
            def __init__(self):
                self.root_block = FakeBlock(graph)
                self.subblocks = {
                    f"masked_subblock{i}": FakeBlock(subgraph)
                    for i, subgraph in enumerate(subgraphs)
                }
                self.indirect_vars = list(indirect_vars)
                self.indexing_exprs = indexing_exprs
                self._bounds = FakeBounds()

            def bounds(self):
                return self._bounds

        return FakeLoopBody()

    def _make_argreduce_loop_body(self, logical_index):
        r0, r1 = sympy.symbols("r0 r1", integer=True, nonnegative=True)

        def fn(index, reduction_index):
            value = V.ops.constant(1.0, torch.float32)
            index = V.ops.index_expr(logical_index(*reduction_index), torch.int64)
            return V.ops.reduction(torch.int64, torch.float32, "argmax", (value, index))

        with (
            config.patch(constant_and_index_propagation=False),
            V.set_graph_handler(_FakeGraph()),
        ):
            loop_body = LoopBody(
                fn,
                ([], [r0, r1]),
                {r0: 4, r1: 8},
                [],
                [r0, r1],
            )
        return loop_body

    @staticmethod
    def _argreduce_nodes(loop_body):
        graph = loop_body.root_block.graph
        reduction = graph.find_nodes(op="call_method", target="reduction")[0]
        value, index_expr = reduction.args[-1]
        get_index = index_expr.args[1]
        return graph, reduction, value, index_expr, get_index

    def test_remove_redundant_argreduce_index(self):
        loop_body = self._make_argreduce_loop_body(lambda r0, r1: 8 * r0 + r1)
        graph, reduction, value, index_expr, get_index = self._argreduce_nodes(
            loop_body
        )
        with V.set_graph_handler(_FakeGraph()):
            remove_redundant_argreduce_indices([loop_body])

        self.assertIs(reduction.args[-1], value)
        self.assertIn(index_expr, graph.nodes)
        self.assertIn(get_index, graph.nodes)

    def test_keep_non_native_argreduce_index(self):
        loop_body = self._make_argreduce_loop_body(lambda r0, r1: 4 * r1 + r0)
        graph, reduction, _, index_expr, get_index = self._argreduce_nodes(loop_body)
        with V.set_graph_handler(_FakeGraph()):
            remove_redundant_argreduce_indices([loop_body])

        self.assertIs(reduction.args[-1][1], index_expr)
        self.assertIn(index_expr, graph.nodes)
        self.assertIn(get_index, graph.nodes)

    def test_keep_shared_non_native_argreduce_index(self):
        original = self._make_argreduce_loop_body(lambda r0, r1: 8 * r0 + r1)
        r0, r1 = original.reduce_vars
        with V.set_graph_handler(_FakeGraph()):
            native = LoopBody(
                original,
                ([], [r0, r1]),
                original.var_ranges,
                [],
                [r0, r1],
                allow_same_symbol_in_index=True,
            )
            reordered = LoopBody(
                original,
                ([], [r0, r1]),
                original.var_ranges,
                [],
                [r0, r1],
                allow_same_symbol_in_index=True,
            )
            reordered.indexing_exprs["index0"] = 4 * r1 + r0
            remove_redundant_argreduce_indices([native, reordered])

        self.assertIs(native.root_block.graph, reordered.root_block.graph)
        self.assertIsInstance(self._argreduce_nodes(native)[1].args[-1], tuple)
        self.assertIsInstance(self._argreduce_nodes(reordered)[1].args[-1], tuple)
        self.assertEqual(reordered.indexing_exprs["index0"], 4 * r1 + r0)

    def test_index_expr_mixed_use_converts_in_place(self):
        # When the same index_expr is used both as an index (load) and as a
        # value (add), we convert it in-place to value_expr. The load still
        # works because value_expr is a superset of index_expr semantics.
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        load = graph.call_method("load", (ops, "arg0", index_expr))
        add = graph.call_method("add", (ops, load, index_expr))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, add, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 1),
                load: ValueRanges(0, 1),
                add: ValueRanges(0, 2**40),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "value_expr")
        self.assertEqual(load.args[2], index_expr)
        self.assertEqual(add.args[2], index_expr)

    def test_value_expr_dtype_deduction_uses_requested_dtype(self):
        self.assertEqual(
            deduce_output_dtype_by_name("value_expr", "expr", torch.float64),
            torch.float64,
        )

    def test_index_expr_unknown_value_use_converts_conservatively(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        unknown = graph.call_method("unknown_value_op", (ops, index_expr))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, unknown, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 2**40),
                unknown: ValueRanges(0, 2**40),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "value_expr")

    def test_index_expr_load_index_stays_indexing(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        load = graph.call_method("load", (ops, "arg0", index_expr))
        add = graph.call_method("add", (ops, load, load))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, add, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 2**40),
                load: ValueRanges(0, 2**40),
                add: ValueRanges(0, 2**41),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "index_expr")
        self.assertEqual(load.args[2], index_expr)
        self.assertEqual([], [n for n in graph.nodes if n.target == "value_expr"])

    def test_index_expr_set_indirect_stays_indexing(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        graph.call_module("set_indirect0", (index_expr,))
        load_index = graph.call_module("get_index", ("i1",))
        load = graph.call_method("load", (ops, "arg0", load_index))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, load, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        indirect0 = sympy.Symbol("indirect0", integer=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 2**40),
                load: ValueRanges(0, 1),
            },
            {"i0": i0, "i1": indirect0},
            {i0: ValueRanges(0, 1)},
            indirect_vars=(indirect0,),
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "index_expr")

    def test_index_expr_set_indirect_value_use_converts(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        set_input = graph.call_method("index_expr", (ops, get_index, torch.int64))
        graph.call_module("set_indirect0", (set_input,))
        value_get_index = graph.call_module("get_index", ("i1",))
        value_index_expr = graph.call_method(
            "index_expr", (ops, value_get_index, torch.int64)
        )
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method(
            "store", (ops, "buf0", store_index, value_index_expr, None)
        )
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        indirect0 = sympy.Symbol("indirect0", integer=True)
        loop_body = self._make_loop_body(
            graph,
            {
                set_input: ValueRanges(0, 2**40),
                value_index_expr: ValueRanges(0, 2**40),
            },
            {"i0": i0, "i1": indirect0 + 1},
            {i0: ValueRanges(0, 1)},
            indirect_vars=(indirect0,),
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(set_input.target, "value_expr")
        self.assertEqual(value_index_expr.target, "value_expr")

    def test_index_expr_device_assert_async_stays_indexing(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        cond = graph.call_method("lt", (ops, index_expr, 16))
        assert_result = graph.call_method("device_assert_async", (ops, cond, "msg"))
        store_index = graph.call_module("get_index", ("i0",))
        store_value = graph.call_method("constant", (ops, 0, torch.int64))
        store = graph.call_method(
            "store", (ops, "buf0", store_index, store_value, None)
        )
        graph.output((assert_result, store))

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 2**40),
                cond: ValueRanges.unknown(),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "index_expr")

    def test_index_expr_sort_and_scan_value_sinks(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        sort_index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        scan_index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        sort = graph.call_method(
            "sort",
            (ops, (torch.int64,), (sort_index_expr,), False, False),
        )
        scan = graph.call_module(
            "scan0",
            ((torch.int64,), (scan_index_expr,)),
        )
        graph.output((sort, scan))

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                sort_index_expr: ValueRanges(0, 2**40),
                scan_index_expr: ValueRanges(0, 2**40),
                sort: ValueRanges(0, 2**40),
                scan: ValueRanges(0, 2**40),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(sort.args[2][0].target, "value_expr")
        self.assertEqual(scan.args[1][0].target, "value_expr")

    def test_index_expr_masked_subblock_value_use(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        mask = graph.placeholder("mask")
        other_get_index = graph.call_module("get_index", ("i0",))
        other_index_expr = graph.call_method(
            "index_expr", (ops, other_get_index, torch.int64)
        )
        masked = graph.call_module("masked_subblock0", (mask, other_index_expr))
        graph.output(masked)

        subgraph = Graph()
        sub_ops = subgraph.placeholder("ops")
        get_index = subgraph.call_module("get_index", ("i0",))
        index_expr = subgraph.call_method(
            "index_expr", (sub_ops, get_index, torch.int64)
        )
        subgraph.output(index_expr)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                other_index_expr: ValueRanges(0, 2**40),
                index_expr: ValueRanges(0, 2**40),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
            subgraphs=(subgraph,),
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(masked.args[1].target, "value_expr")
        self.assertEqual(index_expr.target, "value_expr")

    def test_index_expr_float_value_use_preserves_requested_dtype(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.float32))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, index_expr, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 2**31),
            },
            {"i0": 2147483648 * i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "value_expr")
        self.assertEqual(index_expr.args[2], torch.float32)

    def test_index_expr_value_use_preserves_requested_dtype(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, index_expr, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 1),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "value_expr")
        self.assertEqual(index_expr.args[2], torch.int64)

    def test_existing_value_expr_dtype_is_not_rewritten(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        value_expr = graph.call_method("value_expr", (ops, get_index, torch.int64))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        load = graph.call_method("load", (ops, "arg0", index_expr))
        add = graph.call_method("add", (ops, load, value_expr))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, add, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                value_expr: ValueRanges(0, 1),
                index_expr: ValueRanges(0, 1),
                load: ValueRanges(0, 1),
                add: ValueRanges(0, 2),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(value_expr.target, "value_expr")
        self.assertEqual(value_expr.args[2], torch.int64)
        self.assertEqual(index_expr.target, "index_expr")

    def test_index_expr_getitem_value_use_propagates_to_tuple_source(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.float32))
        frexp = graph.call_method("frexp", (ops, index_expr))
        getitem = graph.call_function(operator.getitem, (frexp, 0))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, getitem, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 1),
                frexp: ValueRanges(0, 1),
                getitem: ValueRanges(0, 1),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "value_expr")
        self.assertEqual(index_expr.args[2], torch.float32)

    def test_index_expr_masked_subblock_index_use_stays_indexing(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        mask = graph.placeholder("mask")
        other_get_index = graph.call_module("get_index", ("i0",))
        other_index_expr = graph.call_method(
            "index_expr", (ops, other_get_index, torch.int64)
        )
        masked = graph.call_module("masked_subblock0", (mask, other_index_expr))
        load = graph.call_method("load", (ops, "arg0", masked))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, load, None))
        graph.output(store)

        subgraph = Graph()
        sub_ops = subgraph.placeholder("ops")
        get_index = subgraph.call_module("get_index", ("i0",))
        index_expr = subgraph.call_method(
            "index_expr", (sub_ops, get_index, torch.int64)
        )
        subgraph.output(index_expr)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                other_index_expr: ValueRanges(0, 2**40),
                masked: ValueRanges(0, 2**40),
                load: ValueRanges(0, 1),
                index_expr: ValueRanges(0, 2**40),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
            subgraphs=(subgraph,),
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(other_index_expr.target, "index_expr")
        self.assertEqual(index_expr.target, "index_expr")

    def test_index_expr_masked_subblock_mixed_use_converts_in_place(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        mask = graph.placeholder("mask")
        other_get_index = graph.call_module("get_index", ("i0",))
        other_index_expr = graph.call_method(
            "index_expr", (ops, other_get_index, torch.int64)
        )
        masked = graph.call_module("masked_subblock0", (mask, other_index_expr))
        load = graph.call_method("load", (ops, "arg0", masked))
        add = graph.call_method("add", (ops, load, masked))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, add, None))
        graph.output(store)

        subgraph = Graph()
        sub_ops = subgraph.placeholder("ops")
        get_index = subgraph.call_module("get_index", ("i0",))
        index_expr = subgraph.call_method(
            "index_expr", (sub_ops, get_index, torch.int64)
        )
        subgraph.output(index_expr)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                other_index_expr: ValueRanges(0, 2**40),
                masked: ValueRanges(0, 2**40),
                load: ValueRanges(0, 1),
                add: ValueRanges(0, 2**40),
                index_expr: ValueRanges(0, 2**40),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
            subgraphs=(subgraph,),
        )

        convert_index_expr_to_value_expr(loop_body)

        # The subblock result has both indexing and value uses. We do not clone,
        # so the value use makes both paths use value_expr.
        self.assertEqual(masked.args[1].target, "value_expr")
        self.assertEqual(index_expr.target, "value_expr")


if __name__ == "__main__":
    run_tests()
