import os
import sys
import unittest
from torch.testing._internal.common_utils import GRAPH_EXECUTOR, ProfilingMode, enable_profiling_mode
import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, disable_autodiff_subgraph_inlining

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# NB: torch.jit.script, when used as a function, uses the current scope
# to resolve variable names. This function cannot be made local to
# TestAutodiffSubgraphSlicing because those tests call torch.jit.script on functions
# in a different scope than they are defined in.
@torch.jit.ignore
def pyfn(a, b):
    return a * b

@unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.SIMPLE, "Simple Executor doesn't support gradients")
class TestAutodiffSubgraphSlicing(JitTestCase):
    # TODO: It is better if we can test directly on graphs instead of the current
    # end-to-end fashion.
    def _perform_ad_subgraph_slicing(self, fn, *input_sizes):
        with disable_autodiff_subgraph_inlining():
            with enable_profiling_mode():
                ge = torch.jit.script(fn)
                inputs = [torch.randn(size, requires_grad=True) for size in input_sizes]
                ge(*inputs, profile_and_replay=True)
                return ge.graph_for(*inputs)

    def assertGraphSize(self, graph, size):
        nodes = list(filter(lambda n : n.kind() != "prim::BailOut" and n.kind() != "prim::BailoutTemplate", graph.nodes()))
        self.assertEqual(len(list(nodes)), size)

    def test_chunk_constant_script_ad(self):
        @torch.jit.script
        def func(x):
            x1, x2 = torch.chunk(x, 2)
            return (x1, x2)

        input = torch.rand(6, 10).requires_grad_()
        with disable_autodiff_subgraph_inlining():
            with enable_profiling_mode():
                output = func(input, profile_and_replay=True)
                self.assertAutodiffNode(func.graph_for(input), True, ['prim::ConstantChunk'], [])

    def test_simple_merge(self):
        # o --> o
        def fn(x, y, z):
            a = x * y
            b = a * z
            return b

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)

        self.assertGraphSize(graph, 1)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 1)

    def test_simple_no_merge(self):
        # o: autodiff supported. x: not autodiff supported.
        # o --> x
        def fn(x, y, z):
            a = x * y
            b = pyfn(a, z)
            return b

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)

        self.assertGraphSize(graph, 2)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 1)

    def test_does_not_merge_unrelated(self):
        # o  o
        def fn(w, x, y, z):
            a = x * y
            b = w * z
            return a, b

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1, 1)

        self.assertGraphSize(graph, 3)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 2)

    def test_merges_without_cycles(self):
        # o --> o --> o
        # |           ^
        #  \_________/
        def fn(w, x, y):
            a = w * x
            b = a * y
            c = a * b
            return c

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)

        self.assertGraphSize(graph, 1)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 1)

    def test_merges_dense(self):
        #   o      o
        #   |\    /|
        #   | \  / |
        #   |  /\  |
        #   vv    vv
        #   o      o
        def fn(x, y):
            a, b = x.chunk(2)
            c, d = y.chunk(2)
            return a + c, b + d

        graph = self._perform_ad_subgraph_slicing(fn, 2, 2)

        self.assertGraphSize(graph, 2)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 1)

    def test_does_not_create_cycles(self):
        # o --> x --> o
        # |           ^
        #  \_________/
        def fn(w, x, y):
            a = w * x
            b = pyfn(a, y)
            c = a * b
            return c

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)

        self.assertGraphSize(graph, 3)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 2)

    def test_merges_up(self):
        # o --> x     o
        # |           ^
        #  \_________/
        def fn(w, x, y, z):
            a = w * x
            b = pyfn(a, y)
            c = a * z
            return b, c

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1, 1)

        self.assertGraphSize(graph, 3)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 1)

    def test_merges_down(self):
        # o     x --> o
        # |           ^
        #  \_________/
        def fn(v, w, x, y):
            a = v * w
            b = pyfn(x, y)
            c = b * a
            return a, c

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1, 1)

        # GuardElimination can't get rid of a prim::BailOut on ^pyfn
        # which makes us create two `prim::DifferentiableGraph`s
        # instead of just one
        num_nodes = 4 if GRAPH_EXECUTOR == ProfilingMode.PROFILING else 3
        self.assertGraphSize(graph, num_nodes)
        num_diff_nodes = 2 if GRAPH_EXECUTOR == ProfilingMode.PROFILING else 1
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', num_diff_nodes)

    def test_respects_lexical_scoping(self):
        def fn(x, k):
            y = x * 1.1
            if bool(k):
                k = k + y
            z = y * k
            return z, k

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1)

        # We should not have combined the two multiplications into
        # the same group; they should each be a separate DiffGraph
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 2)
