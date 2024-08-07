# Owner(s): ["module: fx"]

import os
import sys
from typing import Callable

import torch
import torch.nn.functional as F
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx


pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
import unittest

from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests
from torch.testing._internal.jit_utils import JitTestCase


class WrapperModule(torch.nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class TestMatcher(JitTestCase):
    def test_subgraph_matcher_with_attributes(self):
        class LargeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._weight = torch.nn.Parameter(torch.ones(3, 3))
                self._bias = torch.nn.Parameter(torch.ones(3, 3))

            def forward(self, x):
                return torch.ops.aten.addmm.default(self._bias, x, self._weight)

        # Large Model graph:
        # opcode         name           target              args                 kwargs
        # -------------  -------------  ------------------  -------------------  --------
        # placeholder    x              x                   ()                   {}
        # get_attr       _bias          _bias               ()                   {}
        # get_attr       _weight        _weight             ()                   {}
        # call_function  addmm_default  aten.addmm.default  (_bias, x, _weight)  {}
        # output         output         output              (addmm_default,)     {}
        large_model_graph = symbolic_trace(LargeModel()).graph

        class PatternModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._weight_1 = torch.nn.Parameter(torch.ones(5, 5))
                self._bias_1 = torch.nn.Parameter(torch.ones(5, 5))

            def forward(self, x):
                return torch.ops.aten.addmm.default(self._bias_1, x, self._weight_1)

        pattern_graph = torch.fx.symbolic_trace(PatternModel()).graph

        subgraph_matcher = SubgraphMatcher(pattern_graph)
        match_result = subgraph_matcher.match(large_model_graph)
        self.assertEqual(len(match_result), 1)

    def test_subgraph_matcher_with_list(self):
        def original(x, y):
            return torch.ops.aten.view(x, [5, y.shape[0]])

        original_graph = torch.fx.symbolic_trace(original).graph

        def pattern(x, y, z):
            return torch.ops.aten.view(x, [z, y.shape[0]])

        pattern_graph = torch.fx.symbolic_trace(pattern).graph

        subgraph_matcher = SubgraphMatcher(pattern_graph)
        match_result = subgraph_matcher.match(original_graph)
        self.assertEqual(len(match_result), 1)

    def test_subgraph_matcher_with_list_bad(self):
        def original(x, y):
            return torch.ops.aten._reshape_alias_copy.default(
                x, [1, y.shape[0]], [y.shape[1], y.shape[1]]
            )

        original_graph = torch.fx.symbolic_trace(original).graph

        def pattern(x, y, b):
            return torch.ops.aten._reshape_alias_copy.default(
                x, [b, y.shape[0], y.shape[1]], [y.shape[1]]
            )

        pattern_graph = torch.fx.symbolic_trace(pattern).graph

        subgraph_matcher = SubgraphMatcher(pattern_graph)
        match_result = subgraph_matcher.match(original_graph)
        self.assertEqual(len(match_result), 0)

    def test_subgraph_matcher_ignore_literals(self):
        def original(x):
            return x + 1

        original_graph = make_fx(original)(torch.ones(3, 3)).graph
        original_graph.eliminate_dead_code()

        def pattern(x):
            return x + 2

        pattern_graph = make_fx(pattern)(torch.ones(4, 4)).graph
        pattern_graph.eliminate_dead_code()

        subgraph_matcher = SubgraphMatcher(pattern_graph)
        match_result = subgraph_matcher.match(original_graph)
        self.assertEqual(len(match_result), 0)

        subgraph_matcher = SubgraphMatcher(pattern_graph, ignore_literals=True)
        match_result = subgraph_matcher.match(original_graph)
        self.assertEqual(len(match_result), 1)

    def test_variatic_arg_matching(self):
        inputs = (torch.randn(20, 16, 50, 32),)

        def maxpool(x, kernel_size, stride, padding, dilation):
            return torch.ops.aten.max_pool2d_with_indices.default(
                x, kernel_size, stride, padding, dilation
            )

        maxpool_graph = torch.fx.symbolic_trace(maxpool).graph

        maxpool_matcher = SubgraphMatcher(maxpool_graph)
        match_result = maxpool_matcher.match(maxpool_graph)
        self.assertEqual(len(match_result), 1)

        # Graph only contains "stride" argument
        maxpool_s = torch.nn.MaxPool2d(kernel_size=2, stride=1).eval()
        maxpool_s_graph = make_fx(maxpool_s)(*inputs).graph
        match_s_result = maxpool_matcher.match(maxpool_s_graph)
        self.assertEqual(len(match_s_result), 1)

        # Graph only contains "padding" argument
        maxpool_p = torch.nn.MaxPool2d(kernel_size=2, padding=1)
        maxpool_p_graph = make_fx(maxpool_p)(*inputs).graph
        match_p_result = maxpool_matcher.match(maxpool_p_graph)
        self.assertEqual(len(match_p_result), 1)

        # Graph only contains "stride, padding" argument
        maxpool_sp = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        maxpool_sp_graph = make_fx(maxpool_sp)(*inputs).graph
        match_sp_result = maxpool_matcher.match(maxpool_sp_graph)
        self.assertEqual(len(match_sp_result), 1)

    @unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
    def test_split_to_graph_and_name_node_map(self):
        """Testing the internal helper function for splitting the pattern graph"""
        from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
            _split_to_graph_and_name_node_map,
        )

        def pattern(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu_mul_by_two = relu * 2
            return relu, relu_mul_by_two, {"conv": conv, "relu": relu}

        from torch._export import capture_pre_autograd_graph

        example_inputs = (
            torch.randn(1, 3, 3, 3) * 10,
            torch.randn(3, 3, 3, 3),
        )
        pattern_gm = capture_pre_autograd_graph(WrapperModule(pattern), example_inputs)
        before_split_res = pattern_gm(*example_inputs)
        pattern_gm, name_node_map = _split_to_graph_and_name_node_map(pattern_gm)
        after_split_res = pattern_gm(*example_inputs)
        self.assertEqual(before_split_res[0], after_split_res[0])
        self.assertEqual(before_split_res[1], after_split_res[1])

    @unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
    def test_matcher_with_name_node_map_function(self):
        """Testing SubgraphMatcherWithNameNodeMap with function pattern"""

        def target_graph(x, weight):
            x = x * 2
            weight = weight * 3
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu2 = relu * 2
            return relu + relu2

        def pattern(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu_mul_by_two = relu * 2
            return relu, relu_mul_by_two, {"conv": conv, "relu": relu}

        from torch._export import capture_pre_autograd_graph

        example_inputs = (
            torch.randn(1, 3, 3, 3) * 10,
            torch.randn(3, 3, 3, 3),
        )
        pattern_gm = capture_pre_autograd_graph(WrapperModule(pattern), example_inputs)
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        target_gm = capture_pre_autograd_graph(
            WrapperModule(target_graph), example_inputs
        )
        internal_matches = matcher.match(target_gm.graph)
        for internal_match in internal_matches:
            name_node_map = internal_match.name_node_map
            assert "conv" in name_node_map
            assert "relu" in name_node_map
            name_node_map["conv"].meta["custom_annotation"] = "annotation"
            # check if we correctly annotated the target graph module
            for n in target_gm.graph.nodes:
                if n == name_node_map["conv"]:
                    assert (
                        "custom_annotation" in n.meta
                        and n.meta["custom_annotation"] == "annotation"
                    )

    @unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
    def test_matcher_with_name_node_map_module(self):
        """Testing SubgraphMatcherWithNameNodeMap with module pattern"""

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        class Pattern(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                linear = self.linear(x)
                # Note: we can't put "weight": self.linear.weight in dictionary since
                # nn.Parameter is not an allowed output type in dynamo
                return linear, {"linear": linear, "x": x}

        from torch._export import capture_pre_autograd_graph

        example_inputs = (torch.randn(3, 5),)
        pattern_gm = capture_pre_autograd_graph(Pattern(), example_inputs)
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        target_gm = capture_pre_autograd_graph(M(), example_inputs)
        internal_matches = matcher.match(target_gm.graph)
        for internal_match in internal_matches:
            name_node_map = internal_match.name_node_map
            assert "linear" in name_node_map
            assert "x" in name_node_map
            name_node_map["linear"].meta["custom_annotation"] = "annotation"
            # check if we correctly annotated the target graph module
            for n in target_gm.graph.nodes:
                if n == name_node_map["linear"]:
                    assert (
                        "custom_annotation" in n.meta
                        and n.meta["custom_annotation"] == "annotation"
                    )


if __name__ == "__main__":
    run_tests()
