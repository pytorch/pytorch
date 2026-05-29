#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import operator

import torch
import torch.fx
from torch._higher_order_ops.effects import with_effects
from torch.library import Library
from torch.testing._internal.common_utils import run_tests, TestCase


logger = logging.getLogger(__name__)

_test_lib = Library("torchcomms", "FRAGMENT")  # noqa: SCOPED_LIBRARY


def _register_mock_op():
    try:
        _test_lib.define("mock_op(Tensor x) -> Tensor")

        @torch.library.impl(_test_lib, "mock_op", "CPU")
        def mock_op_cpu(x):
            return x * 2

        @torch.library.impl(_test_lib, "mock_op", "Meta")
        def mock_op_meta(x):
            return torch.empty_like(x)

    except RuntimeError:
        pass


_register_mock_op()


def _count_nodes_by_pattern(gm: torch.fx.GraphModule, pattern: str) -> int:
    count = 0
    for node in gm.graph.nodes:
        if node.op == "call_function":
            target_name = str(node.target)
            if pattern in target_name:
                count += 1
    return count


def _has_node_with_pattern(gm: torch.fx.GraphModule, pattern: str) -> bool:
    return _count_nodes_by_pattern(gm, pattern) > 0


def _get_output_node(gm: torch.fx.GraphModule):
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "output":
            return node
    return None


class TestReinplacementPass(TestCase):
    def test_reinplacement_basic(self):
        from torch.comms.functional.passes import reinplacement_pass

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                y = x.add(1)
                return y

        model = SimpleModule()
        example_input = torch.randn(4, 4)

        gm = torch.fx.symbolic_trace(model)
        result_gm = reinplacement_pass(gm)

        expected = model(example_input)
        actual = result_gm(example_input)
        torch.testing.assert_close(actual, expected)

    def test_reinplacement_preserves_semantics(self):
        from torch.comms.functional.passes import reinplacement_pass

        class ComplexModule(torch.nn.Module):
            def forward(self, x, y):
                a = x + y
                b = a * 2
                c = b - 1
                return c

        model = ComplexModule()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        gm = torch.fx.symbolic_trace(model)
        result_gm = reinplacement_pass(gm)

        expected = model(x, y)
        actual = result_gm(x, y)
        torch.testing.assert_close(actual, expected)


class TestStripWithEffectsPass(TestCase):
    def test_strip_with_effects_no_op_on_empty_graph(self):
        from torch.comms.functional.passes import strip_with_effects_pass

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(x)

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        result_gm = strip_with_effects_pass(gm)

        test_input = torch.randn(4, 4)
        result = result_gm(test_input)
        torch.testing.assert_close(result, test_input)

    def test_strip_with_effects_removes_torchcomms_wrapper(self):
        from torch.comms.functional.passes import strip_with_effects_pass

        graph = torch.fx.Graph()
        x = graph.placeholder("x")

        initial_token = graph.call_function(torch.ops.aten._make_dep_token.default, ())

        with_effects_node = graph.call_function(
            with_effects,
            (initial_token, torch.ops.torchcomms.mock_op.default, x),
        )

        result = graph.call_function(operator.getitem, (with_effects_node, 1))
        graph.output(result)

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        self.assertTrue(_has_node_with_pattern(gm, "with_effects"))

        result_gm = strip_with_effects_pass(gm)

        self.assertFalse(_has_node_with_pattern(result_gm, "with_effects"))

        self.assertTrue(_has_node_with_pattern(result_gm, "torchcomms.mock_op"))

    def test_strip_with_effects_handles_token_in_output(self):
        from torch.comms.functional.passes import strip_with_effects_pass

        graph = torch.fx.Graph()
        x = graph.placeholder("x")

        initial_token = graph.call_function(torch.ops.aten._make_dep_token.default, ())

        with_effects_node = graph.call_function(
            with_effects,
            (initial_token, torch.ops.torchcomms.mock_op.default, x),
        )

        token = graph.call_function(operator.getitem, (with_effects_node, 0))
        result = graph.call_function(operator.getitem, (with_effects_node, 1))

        graph.output((token, result))

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        result_gm = strip_with_effects_pass(gm)

        self.assertFalse(_has_node_with_pattern(result_gm, "with_effects"))

        self.assertFalse(_has_node_with_pattern(result_gm, "getitem"))

        output_node = _get_output_node(result_gm)
        self.assertIsNotNone(output_node)
        output_tuple = output_node.args[0]
        self.assertEqual(len(output_tuple), 2)

    def test_strip_with_effects_erases_nodes(self):
        from torch.comms.functional.passes import strip_with_effects_pass

        graph = torch.fx.Graph()
        x = graph.placeholder("x")

        initial_token = graph.call_function(torch.ops.aten._make_dep_token.default, ())

        with_effects_node = graph.call_function(
            with_effects,
            (initial_token, torch.ops.torchcomms.mock_op.default, x),
        )

        result = graph.call_function(operator.getitem, (with_effects_node, 1))
        graph.output(result)

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        nodes_before = len(list(gm.graph.nodes))

        result_gm = strip_with_effects_pass(gm)

        nodes_after = len(list(result_gm.graph.nodes))

        self.assertLess(nodes_after, nodes_before)

    def test_strip_with_effects_skips_non_torchcomms(self):
        from torch.comms.functional.passes import strip_with_effects_pass

        graph = torch.fx.Graph()
        x = graph.placeholder("x")

        initial_token = graph.call_function(torch.ops.aten._make_dep_token.default, ())

        with_effects_node = graph.call_function(
            with_effects,
            (initial_token, torch.ops.aten.add.Tensor, x, x),
        )

        result = graph.call_function(operator.getitem, (with_effects_node, 1))
        graph.output(result)

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        result_gm = strip_with_effects_pass(gm)

        self.assertTrue(_has_node_with_pattern(result_gm, "with_effects"))


class TestPassesIntegration(TestCase):
    def test_passes_can_be_chained(self):
        from torch.comms.functional.passes import (
            reinplacement_pass,
            strip_with_effects_pass,
        )

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = SimpleModule()
        gm = torch.fx.symbolic_trace(model)

        gm = strip_with_effects_pass(gm)
        gm = reinplacement_pass(gm)

        test_input = torch.randn(4, 4)
        expected = model(test_input)
        actual = gm(test_input)
        torch.testing.assert_close(actual, expected)

    def test_passes_handle_empty_graph(self):
        from torch.comms.functional.passes import (
            reinplacement_pass,
            strip_with_effects_pass,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(x)

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        gm = strip_with_effects_pass(gm)
        gm = reinplacement_pass(gm)

        test_input = torch.randn(4, 4)
        result = gm(test_input)
        torch.testing.assert_close(result, test_input)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_tests()
