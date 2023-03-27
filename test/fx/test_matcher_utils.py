# Owner(s): ["module: fx"]

import os
import sys

import torch
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher, replace_literals_with_placeholders
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


    def check_replace_literals(self, f, inputs, expected_num_placeholders):
        gm = make_fx(f)(*inputs)
        graph = replace_literals_with_placeholders(gm.graph)

        num_placeholder = 0
        for node in graph.nodes:
            if node.op == "placeholder":
                num_placeholder += 1
        self.assertEqual(num_placeholder, expected_num_placeholders)

    def test_replace_literals_regular(self):
        """
        Tests replacing literals with placeholders in the case where there are
        no literals to replace.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        inputs = (torch.randn(3, 5), torch.randn(3, 5))
        self.check_replace_literals(M(), inputs, 2)

    def test_replace_literals_args(self):
        """
        Tests replacing literals with placeholders in the case where args are
        literals
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3)

            def forward(self, x):
                return self.conv(x)

        inputs = (torch.randn(1, 3, 256, 256),)
        self.check_replace_literals(M(), inputs, 7)

    def test_replace_literals_kwargs(self):
        """
        Tests replacing literals with placeholders in the case where kwargs are
        literals. This also contains a `getattr` node which should not be
        modified.
        """

        def f(x: torch.Tensor, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
            return torch.addbmm(x, batch1, batch2, alpha=2, beta=3)

        inputs = (torch.randn(3, 5), torch.randn(10, 3, 4), torch.randn(10, 4, 5))
        self.check_replace_literals(f, inputs, 5)

    def test_replace_literals_linear(self):
        """
        Tests replacing literals with placeholders in the case there are
        `getitem` calls which do not have a schema.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.split(x, 10)

        inputs = (torch.randn(10),)
        self.check_replace_literals(M(), inputs, 3)
