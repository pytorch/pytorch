# Owner(s): ["module: fx"]

import os
import sys

import torch
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.testing._internal.jit_utils import JitTestCase

class TestMatcher(JitTestCase):
    def test_subgraph_matcher_with_attributes(self):
        class LargeModel(torch.nn.Module):
            def __init__(self):
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
            def __init__(self):
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
            return torch.ops.aten._reshape_alias_copy.default(x, [1, y.shape[0]], [y.shape[1], y.shape[1]])
        original_graph = torch.fx.symbolic_trace(original).graph

        def pattern(x, y, b):
            return torch.ops.aten._reshape_alias_copy.default(x, [b, y.shape[0], y.shape[1]], [y.shape[1]])
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
            return torch.ops.aten.max_pool2d_with_indices.default(x, kernel_size, stride, padding, dilation)
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
