import os
import re
import sys

import torch
from torch.fx import symbolic_trace, subgraph_rewriter

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_fx.py TESTNAME\n\n"
                       "instead.")

class TestSubgraphRewriter(JitTestCase):

    def test_subgraph_rewriter_preserves_logic(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w1, w2, b1, b2):
                m1 = torch.cat([w1, w2])
                m2 = torch.cat([x, b2])
                t1 = torch.sum(w1, 1)
                t2 = torch.addmm(b1, m1, m2.t())
                return torch.eq(torch.sum(t1), torch.sum(t2))

        def comparison(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            t1 = torch.sum(w1, 1)
            t2 = torch.addmm(b1, m1, m2.t())
            return torch.eq(torch.sum(t1), torch.sum(t2))

        def pattern(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            return torch.addmm(b1, m1, m2.t())

        traced_module = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)
        w1 = torch.rand(1, 3)
        w2 = torch.rand(1, 3)
        b1 = torch.rand(2, 2)
        b2 = torch.rand(1, 3)

        # Replace `pattern` with the same pattern (shouldn't change
        # the underlying logic)
        subgraph_rewriter.replace_pattern(traced_module, pattern, pattern)

        traced_module.graph.lint(traced_module)

        ref_outs = comparison_fn(x, w1, w2, b1, b2)
        test_outs = traced_module.forward(x, w1, w2, b1, b2)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_with_oneliner_pattern(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w1, w2, b1, b2):
                m1 = torch.cat([w1, w2])
                m2 = torch.cat([x, b2])
                t1 = torch.sum(w1, 1)
                t2 = torch.addmm(b1, m1, m2.t())
                return torch.eq(torch.sum(t1), torch.sum(t2))

        def pattern(x, w1, w2, b1, b2):
            return torch.cat([x, b2])

        def replacement(x, w1, w2, b1, b2):
            return torch.cat([b2, w2])

        traced_module = symbolic_trace(M())
        subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)

        traced_module.graph.lint(traced_module)

    def test_subgraph_rewriter_single_pattern_match(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w1, w2, b1, b2):
                m1 = torch.cat([w1, w2])
                m2 = torch.cat([x, b2])
                t1 = torch.sum(w1, 1)
                t2 = torch.addmm(b1, m1, m2.t())
                return torch.eq(torch.sum(t1), torch.sum(t2))

        def comparison(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            t1 = torch.sum(w1, 1)
            t2 = torch.addmm(b1, m1, m2.t())
            return torch.eq(torch.sum(t1), torch.sum(t2))

        def pattern(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            return torch.addmm(b1, m1, m2.t())

        def replacement(x, w1, w2, b1, b2):
            m2 = torch.cat([x, b2])
            m1 = torch.cat([w1, w2])
            more_lines = torch.mm(w1, w2.t())
            return torch.addmm(b1, m1, m2.t())

        traced_module = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)
        w1 = torch.rand(1, 3)
        w2 = torch.rand(1, 3)
        b1 = torch.rand(2, 2)
        b2 = torch.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)

        # Check that the pattern in `fn` was replaced. The outputs should
        # be the same since the only difference is in code order
        ref_outs = comparison_fn(x, w1, w2, b1, b2)
        test_outs = traced_module.forward(x, w1, w2, b1, b2)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_multiple_pattern_match(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w1, w2, b1, b2):
                m1 = torch.cat([w1, w2])
                m2 = torch.cat([x, b2])
                m3 = torch.cat([w1, w2])
                t1 = torch.addmm(b1, m3, m2.t())
                t2 = torch.addmm(b1, m1, m2.t())
                return torch.eq(torch.sum(t1), torch.sum(t2))

        def comparison(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            m3 = torch.cat([w1, w2])
            t1 = torch.addmm(b1, m3, m2.t())
            t2 = torch.addmm(b1, m1, m2.t())
            return torch.eq(torch.sum(t1), torch.sum(t2))

        def pattern(x, w1, w2, b1, b2):
            return torch.cat([w1, w2])

        def replacement(x, w1, w2, b1, b2):
            return torch.cat([w2, w1])

        traced_module = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)
        w1 = torch.rand(1, 3)
        w2 = torch.rand(1, 3)
        b1 = torch.rand(2, 2)
        b2 = torch.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)

        traced_module.graph.lint(traced_module)

        # Check that the pattern in `fn` was replaced. The outputs should
        # be the same since the only difference is in argument order.
        ref_outs = comparison_fn(x, w1, w2, b1, b2)
        test_outs = traced_module.forward(x, w1, w2, b1, b2)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_add_params(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w1, w2):
                m1 = torch.cat([w1, w2])
                return torch.mm(m1, x.t())

        def pattern(x, w1, w2):
            return torch.cat([w1, w2])

        def replacement(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([b1, b2])
            return torch.cat([m1, m2])

        traced_module = symbolic_trace(M())

        subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)

        traced_module.graph.lint(traced_module)

        # `compile()` represents the function as an anonymous module,
        # which means that `co_argcount` would always be 0. This means
        # we have to check the produced string for equality instead
        code_str = traced_module.graph.python_code("<string>")
        def_line = re.search("def.*", code_str).group()
        self.assertEqual(def_line,
                         "def forward(self, x, w1, w2, b1, b2):")

    def test_subgraph_rewriter_pattern_extends_to_end_of_original_graph(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w1, w2, b1, b2):
                m1 = torch.cat([w1, w2])
                m2 = torch.cat([x, b2])
                t1 = torch.addmm(b1, m1, m2.t())
                t2 = torch.sum(w1, 1)
                t3 = torch.sum(b1, 1)
                return torch.eq(torch.sum(t2), torch.sum(t3))

        def comparison(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            t1 = torch.addmm(b1, m1, m2.t())
            t2 = torch.sum(w1, 1)
            t3 = torch.sum(b1, 1)
            return torch.eq(torch.sum(t2), torch.sum(t3))

        def pattern(x, w1, w2, b1, b2):
            t2 = torch.sum(w1, 1)
            t3 = torch.sum(b1, 1)
            return torch.eq(torch.sum(t2), torch.sum(t3))

        traced_module = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)
        w1 = torch.rand(1, 3)
        w2 = torch.rand(1, 3)
        b1 = torch.rand(2, 2)
        b2 = torch.rand(1, 3)

        # Replace `pattern` with the same pattern (shouldn't change
        # the underlying logic)
        subgraph_rewriter.replace_pattern(traced_module, pattern, pattern)

        traced_module.graph.lint(traced_module)

        ref_outs = comparison_fn(x, w1, w2, b1, b2)
        test_outs = traced_module.forward(x, w1, w2, b1, b2)
        self.assertEqual(ref_outs, test_outs)
