import os
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
        # Function that we want to perform the rewrite on
        def fn(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            t1 = torch.sum(w1, 1)
            t2 = torch.addmm(b1, m1, m2.t())
            return torch.eq(torch.sum(t1), torch.sum(t2))

        # Pattern we want to match in `fn`
        def pattern(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            return torch.addmm(b1, m1, m2.t())

        # Set up
        traced_fn = symbolic_trace(fn)
        comparison_fn = symbolic_trace(fn)

        # Auxiliary testing variables
        x = torch.rand(1, 3)
        w1 = torch.rand(1, 3)
        w2 = torch.rand(1, 3)
        b1 = torch.rand(2, 2)
        b2 = torch.rand(1, 3)

        # Replace `pattern` with the same pattern
        subgraph_rewriter.replace_pattern(traced_fn.graph, pattern, pattern)

        # Check that the graph is still valid after the rewrite
        traced_fn.graph.lint(traced_fn)

        # Ensure that the graph is functionally the same as it was
        # before the rewrite
        ref_outs = comparison_fn(x, w1, w2, b1, b2)
        test_outs = traced_fn(x, w1, w2, b1, b2)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_with_oneliner_pattern(self):
        # Function that we want to perform the rewrite on
        def fn(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            t1 = torch.sum(w1, 1)
            t2 = torch.addmm(b1, m1, m2.t())
            return torch.eq(torch.sum(t1), torch.sum(t2))

        # Pattern we want to match in `fn`
        def pattern(x, w1, w2, b1, b2):
            return torch.cat([x, b2])

        # What we want to replace `pattern` with
        def replacement(x, w1, w2, b1, b2):
            return torch.cat([b2, w2])

        # Set up
        traced_fn = symbolic_trace(fn)
        subgraph_rewriter.replace_pattern(traced_fn.graph, pattern, replacement)

        # Check that the graph is still valid after the rewrite
        traced_fn.graph.lint(traced_fn)

    def test_subgraph_rewriter_single_pattern_match(self):
        # Function that we want to perform the rewrite on
        def fn(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            t1 = torch.sum(w1, 1)
            t2 = torch.addmm(b1, m1, m2.t())
            return torch.eq(torch.sum(t1), torch.sum(t2))

        # Pattern we want to match in `fn`
        def pattern(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            return torch.addmm(b1, m1, m2.t())

        # What we want to replace `pattern` with
        def replacement(x, w1, w2, b1, b2):
            m2 = torch.cat([x, b2])
            m1 = torch.cat([w1, w2])
            more_lines = torch.mm(w1, w2.t())
            return torch.addmm(b1, m1, m2.t())

        # Set up
        traced_fn = symbolic_trace(fn)
        comparison_fn = symbolic_trace(fn)

        # Auxiliary testing variables
        x = torch.rand(1, 3)
        w1 = torch.rand(1, 3)
        w2 = torch.rand(1, 3)
        b1 = torch.rand(2, 2)
        b2 = torch.rand(1, 3)

        # Replace `pattern` with the replacement
        subgraph_rewriter.replace_pattern(traced_fn.graph, pattern, replacement)

        # Check that the pattern in `fn` was replaced. The outputs should
        # be the same since the only difference is in code order
        ref_outs = comparison_fn(x, w1, w2, b1, b2)
        test_outs = traced_fn(x, w1, w2, b1, b2)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_multiple_pattern_match(self):
        # Function that we want to perform the rewrite on
        def fn(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            m1 = torch.cat([w1, w2])
            t1 = torch.sum(w1, 1)
            t2 = torch.addmm(b1, m1, m2.t())
            return torch.eq(torch.sum(t1), torch.sum(t2))

        # Pattern we want to match in `fn`
        def pattern(x, w1, w2, b1, b2):
            return torch.cat([w1, w2])

        # What we want to replace `pattern` with
        def replacement(x, w1, w2, b1, b2):
            return torch.cat([w2, w1])

        # Set up
        traced_fn = symbolic_trace(fn)
        comparison_fn = symbolic_trace(fn)

        # Auxiliary testing variables
        x = torch.rand(1, 3)
        w1 = torch.rand(1, 3)
        w2 = torch.rand(1, 3)
        b1 = torch.rand(2, 2)
        b2 = torch.rand(1, 3)

        # Replace `pattern` with the replacement
        subgraph_rewriter.replace_pattern(traced_fn.graph, pattern, replacement)

        # Check that the graph is still valid after the rewrite
        traced_fn.graph.lint(traced_fn)

        # Check that the pattern in `fn` was replaced. The outputs should
        # be the same since the only difference is in code order.
        ref_outs = comparison_fn(x, w1, w2, b1, b2)
        test_outs = traced_fn(x, w1, w2, b1, b2)
        self.assertEqual(ref_outs, test_outs)


    def test_subgraph_rewriter_add_params(self):
        # Function that we want to perform the rewrite on
        def fn(x, w1, w2):
            m1 = torch.cat([w1, w2])
            t2 = torch.cat([w1, w1], 1)
            return torch.mm(m1, t2)

        def pattern(x, w1, w2):
            return torch.cat([w1, w2])

        def replacement(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([b1, b2])
            return torch.cat([m1, m2])

        # Set up
        traced_fn = symbolic_trace(fn)
        comparison_fn = symbolic_trace(fn)

        # Replace `pattern` with the replacement
        subgraph_rewriter.replace_pattern(traced_fn.graph, pattern, replacement)

        # Check that the graph is still valid after the rewrite
        traced_fn.graph.lint(traced_fn)
