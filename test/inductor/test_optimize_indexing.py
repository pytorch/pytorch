# Owner(s): ["module: inductor"]

import operator
from unittest import mock

import sympy

import torch
import torch._inductor.optimize_indexing as optimize_indexing
from torch._inductor.codegen.common import deduce_output_dtype_by_name
from torch._inductor.optimize_indexing import convert_index_expr_to_value_expr
from torch.fx import Graph
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._sympy.value_ranges import ValueRanges


class TestOptimizeIndexing(TestCase):
    @staticmethod
    def _make_loop_body(graph, bounds, indexing_exprs, replacement_vals, subgraphs=()):
        class FakeBounds:
            def __init__(self):
                self.replacement_vals = replacement_vals

            def get_bounds(self):
                return bounds

        class FakeBlock:
            def __init__(self, graph):
                self.graph = graph

        class FakeLoopBody:
            indirect_vars = []

            def __init__(self):
                self.root_block = FakeBlock(graph)
                self.subblocks = {
                    f"masked_subblock{i}": FakeBlock(subgraph)
                    for i, subgraph in enumerate(subgraphs)
                }
                self.indexing_exprs = indexing_exprs
                self._bounds = FakeBounds()

            def bounds(self):
                return self._bounds

        return FakeLoopBody()

    def test_index_expr_mixed_use_clones_value_path(self):
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

        value_exprs = [n for n in graph.nodes if n.target == "value_expr"]
        self.assertEqual(len(value_exprs), 1)
        value_expr = value_exprs[0]
        self.assertEqual(index_expr.target, "index_expr")
        self.assertEqual(load.args[2], index_expr)
        self.assertEqual(add.args[2], value_expr)
        self.assertEqual(value_expr.args[2], torch.int64)

    def test_value_expr_dtype_deduction_uses_requested_dtype(self):
        self.assertEqual(
            deduce_output_dtype_by_name("value_expr", "expr", torch.float64),
            torch.float64,
        )

    def test_index_expr_unknown_value_use_stays_indexing(self):
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

        self.assertEqual(index_expr.target, "index_expr")
        self.assertEqual(unknown.args[1], index_expr)
        self.assertEqual([], [n for n in graph.nodes if n.target == "value_expr"])

    def test_index_expr_load_barrier_stays_indexing(self):
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

    def test_index_expr_masked_subblock_mixed_use_prefers_value(self):
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

        self.assertEqual(masked.args[1].target, "value_expr")
        output_node = next(n for n in subgraph.nodes if n.op == "output")
        self.assertEqual(output_node.args[0].target, "value_expr")

    def test_mixed_value_op_clones_value_path(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        add = graph.call_method("add", (ops, index_expr, index_expr))
        load = graph.call_method("load", (ops, "arg0", add))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, add, None))
        graph.output((load, store))

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 1),
                add: ValueRanges(0, 2),
                load: ValueRanges(0, 1),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        value_adds = [
            n
            for n in graph.nodes
            if n.target == "add" and any(arg.target == "value_expr" for arg in n.args)
        ]
        self.assertEqual(len(value_adds), 1)
        value_add = value_adds[0]
        self.assertEqual(load.args[2], add)
        self.assertEqual(store.args[3], value_add)
        self.assertEqual(add.args[1], index_expr)
        self.assertEqual(add.args[2], index_expr)
        self.assertEqual(index_expr.target, "index_expr")

    def test_shared_value_only_ancestors_are_rewritten_once(self):
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        bounds = {index_expr: ValueRanges(0, 1)}

        value = index_expr
        for _ in range(14):
            value = graph.call_method("add", (ops, value, value))
            bounds[value] = ValueRanges(0, 1)
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, value, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            bounds,
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        original_map_arg = optimize_indexing.map_arg
        map_arg_calls = 0

        def counting_map_arg(*args, **kwargs):
            nonlocal map_arg_calls
            map_arg_calls += 1
            return original_map_arg(*args, **kwargs)

        with mock.patch.object(
            optimize_indexing, "map_arg", side_effect=counting_map_arg
        ):
            convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "value_expr")
        self.assertLess(map_arg_calls, 200)


if __name__ == "__main__":
    run_tests()
