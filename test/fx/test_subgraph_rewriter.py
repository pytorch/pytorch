# Owner(s): ["module: fx"]

import operator
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest import mock

import torch
from torch._library.simple_registry import singleton as _simple_registry
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import subgraph_rewriter, symbolic_trace
from torch.fx.annotate import annotate

# Make the helper files in test/ importable
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._python_dispatch import TorchDispatchMode


pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import skipIfTorchDynamo
from torch.testing._internal.jit_utils import JitTestCase


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_fx.py TESTNAME\n\n"
        "instead."
    )


@torch.fx.wrap
def wrapped_gemm_bias_mul(a, b, bias):
    lin_res = torch.nn.functional.linear(a, b, bias=bias)
    mul_res = lin_res * a
    return lin_res, mul_res


@torch.fx.wrap
def wrapped_gemm_bias_mul_with_c(a, b, bias, c):
    lin_res = torch.nn.functional.linear(a, b, bias=bias)
    mul_res = lin_res * c
    return lin_res, mul_res


_side_effect_replacement_call_count = 0


@torch.fx.wrap
def side_effect_replacement(x):
    global _side_effect_replacement_call_count
    _side_effect_replacement_call_count += 1
    return x


class _ReluModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.relu.default(x)


def _relu_pattern(x):
    return torch.ops.aten.relu.default(x)


class _NegModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.neg.default(x)


def _neg_pattern(x):
    return torch.ops.aten.neg.default(x)


class _AddModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.aten.add.Tensor(x, y)


def _add_pattern(x, y):
    return torch.ops.aten.add.Tensor(x, y)


class TestSubgraphRewriter(JitTestCase):
    def _node_with_target(self, gm, target):
        return next(node for node in gm.graph.nodes if node.target == target)

    def _replace_once(self, gm, pattern, replacement):
        matches = subgraph_rewriter.replace_pattern(gm, pattern, replacement)
        self.assertEqual(len(matches), 1)

    def _export_relu(self):
        return torch.export.export(_ReluModel(), (torch.randn(2, 3),)).graph_module

    def _reset_side_effect_replacement_call_count(self):
        global _side_effect_replacement_call_count
        _side_effect_replacement_call_count = 0

    def test_subgraph_rewriter_preserves_logic(self):
        class M(torch.nn.Module):
            def forward(self, x):
                val = torch.neg(x) + torch.relu(x)
                return torch.add(val, val)

        def pattern(x):
            return torch.neg(x) + torch.relu(x)

        def comparison(x):
            val = torch.neg(x) + torch.relu(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)

        # Replace `pattern` with the same pattern (shouldn't change
        # the underlying logic)
        subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_with_oneliner_pattern(self):
        class M(torch.nn.Module):
            def forward(self, x):
                val = torch.neg(x)
                return torch.add(val, val)

        def pattern(x):
            return torch.neg(x)

        def replacement(x):
            return torch.relu(x)

        def comparison(x):
            val = torch.relu(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_with_trivial_replacement(self):
        class M(torch.nn.Module):
            def forward(self, x):
                val = torch.neg(x)
                val = torch.add(val, val)
                return torch.add(val, val)

        def pattern(x):
            return torch.add(x, x)

        def replacement(x):
            return x

        def comparison(x):
            return torch.neg(x)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(1, 5)

        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced, pattern, replacement, []
        )

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        no_replacements = len(matches) == 2 and len(matches[1].replacements) == 0
        self.assertEqual(ref_output, test_output)
        self.assertTrue(no_replacements)

    def test_subgraph_rewriter_single_pattern_match(self):
        class M(torch.nn.Module):
            def forward(self, x):
                val = torch.neg(x) + torch.relu(x)
                return torch.add(val, val)

        def pattern(x):
            return torch.neg(x) + torch.relu(x)

        def replacement(x):
            return torch.relu(x)

        def comparison(x):
            val = torch.relu(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_multiple_pattern_match(self):
        class M(torch.nn.Module):
            def forward(self, x, w1, w2):
                m1 = torch.cat([w1, w2]).sum()
                m2 = torch.cat([w1, w2]).sum()
                return x + torch.max(m1) + torch.max(m2)

        def pattern(w1, w2):
            return torch.cat([w1, w2]).sum()

        def replacement(w1, w2):
            return torch.stack([w1, w2])

        def comparison(x, w1, w2):
            m1 = torch.stack([w1, w2])
            m2 = torch.stack([w1, w2])
            return x + torch.max(m1) + torch.max(m2)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)
        w1 = torch.rand(1, 3)
        w2 = torch.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x, w1, w2)
        test_outs = traced.forward(x, w1, w2)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_graph_argument_order(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.mm(x, y)

        def pattern(x, y):
            return torch.mm(x, y)

        def comparison(x, y):
            return torch.mm(x, y)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)
        y = torch.randn(4, 5)

        subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        traced.graph.lint()

        ref_outs = comparison_fn(x, y)
        test_outs = traced.forward(x, y)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_correct_output_replacement(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                val = torch.neg(y) + torch.relu(x)
                return torch.add(val, val)

        def pattern(x):
            return torch.relu(x)

        def replacement(x):
            return torch.neg(x)

        def comparison(x, y):
            val = torch.neg(y) + torch.neg(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x, y)
        test_outs = traced.forward(x, y)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_traced_as_callable(self):
        class M(torch.nn.Module):
            def forward(self, x):
                val = torch.neg(x) + torch.relu(x)
                return torch.add(val, val)

        class Pattern(torch.nn.Module):
            def forward(self, x):
                return torch.neg(x) + torch.relu(x)

        class Replacement(torch.nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        def comparison(x):
            val = torch.sigmoid(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        traced_pattern = symbolic_trace(Pattern())
        traced_replacement = symbolic_trace(Replacement())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, traced_pattern, traced_replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_pattern_is_entire_graph(self):
        class M(torch.nn.Module):
            def forward(self, x):
                a = torch.neg(x)
                return torch.add(a, a)

        def pattern(x):
            a = torch.neg(x)
            return torch.add(a, a)

        def replacement(x):
            a = torch.sigmoid(x)
            return torch.cat([a, a])

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(replacement)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_pattern_output_pattern_node_can_have_users_that_are_not_matched(
        self,
    ):
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.relu(x)
                return torch.neg(y) - y

        def pattern(x):
            return torch.relu(x)

        def replacement(x):
            return torch.sigmoid(x)

        def comparison(x):
            y = torch.sigmoid(x)
            return torch.neg(y) - y

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_internal_pattern_nodes_cannot_have_users_that_are_not_matched(
        self,
    ):
        class M(torch.nn.Module):
            def forward(self, x, w1, w2, b1, b2):
                m0 = torch.cat([w1, w2])  # noqa: F841
                m1 = torch.cat([w1, w2])
                m2 = torch.cat([x, b2])
                t0 = torch.addmm(b1, m1, m2.t())  # noqa: F841
                t1 = torch.sum(w1, 1)
                t2 = torch.addmm(b1, m1, m2.t())
                return torch.sum(t1), torch.sum(t2)

        def pattern(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            return torch.addmm(b1, m1, m2.t())

        def replacement(x, w1, w2, b1, b2):
            return torch.cat([x, w1, w2])

        traced = symbolic_trace(M())

        # Result should be [] since no matches can be found
        res = subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        self.assertEqual(res, [])

    def test_subgraph_rewriter_placeholder_matching(self):
        """
        This tests that a placeholder Node can be matched to a Node with
        a different number of input Nodes. In the example below, the
        original traced Module looks like this:

            opcode         target                                                      args                      kwargs
            -------------  ----------------------------------------------------------  ------------------------  --------
            placeholder    x                                                           ()                        {}
            call_function  <built-in function add>                                     (x, 3)                    {}
            call_method    dequantize                                                  (add,)                    {}
            call_function  <built-in method sigmoid of type object at 0x7f7c1f440fe0>  (dequantize,)             {}
            call_method    to                                                          (sigmoid, torch.float16)  {}
            output         output                                                      (to,)                     {}

        while the pattern we want to match looks like this:

            opcode         target                                                      args                      kwargs
            -------------  ----------------------------------------------------------  ------------------------  --------
            placeholder    x                                                           ()                        {}
            call_method    dequantize                                                  (x,)                      {}
            call_function  <built-in method sigmoid of type object at 0x7f7c1f440fe0>  (dequantize,)             {}
            call_method    to                                                          (sigmoid, torch.float16)  {}
            output         output                                                      (to,)                     {}

        Here, we want to be able to match the original graph's
        `call_function.add` Node with the pattern graph's
        `placeholder.x` Node.

        Credit to Jerry Zhang (GitHub: jerryzh168) for this test case
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dtype = torch.float16

            def forward(self, x):
                x += 3
                x = x.dequantize()
                x = torch.sigmoid(x)
                dtype = self.dtype
                x = x.to(dtype)
                return x

        def pattern(x):
            x = x.dequantize()
            x = torch.sigmoid(x)
            x = x.to(torch.float16)
            return x

        def replacement(x):
            return x

        def comparison(x):
            return x + 3

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_replaces_referenced_submodules(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                x = x + 1
                return self.submod(self.sigmoid(x))

        class Pattern(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                return self.submod(self.sigmoid(x))

        class Replacement(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tanh = torch.nn.Tanh()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                return self.submod(self.tanh(x))

        class Comparison(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tanh = torch.nn.Tanh()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                x = x + 1
                return self.submod(self.tanh(x))

        traced = symbolic_trace(M())
        comparison = Comparison()

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, Pattern(), Replacement())

        traced.graph.lint()

        ref_outs = comparison(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

        traced.get_submodule("tanh")
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            traced.get_submodule("sigmoid")

        submod = traced.get_submodule("submod")
        self.assertEqual(type(submod), torch.nn.ReLU)

    def test_subgraph_rewriter_annotations_int(self):
        class M1(torch.nn.Module):
            def forward(self, x):
                y: int = x
                return torch.add(x, y)

        class M2(torch.nn.Module):
            def forward(self, x):
                y = annotate(x, int)
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(M1())

        module = M2()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        for n, m in zip(symbolic_traced.graph.nodes, graph.nodes):
            if n.op == "placeholder":
                if n.type is not int:
                    raise AssertionError(f"Expected n.type to be int, got {n.type}")
                if m.type is not int:
                    raise AssertionError(f"Expected m.type to be int, got {m.type}")

    def test_subgraph_rewriter_replace_consecutive_submodules(self):
        def f(x):
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            return torch.sigmoid(x)

        def pattern(x):
            return torch.sigmoid(x)

        def replacement(x):
            return torch.exp(x)

        def comparison(x):
            x = torch.exp(x)
            x = torch.exp(x)
            return torch.exp(x)

        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_with_overlapping_matches(self):
        def f(x):
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            return torch.sigmoid(x)

        def pattern(x):
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            return x

        def replacement(x):
            return torch.neg(x)

        def comparison(x):
            x = torch.neg(x)
            return torch.neg(x)

        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_replace_with_multiple_outputs(self):
        def f(x):
            y = torch.sigmoid(x)
            z = torch.relu(x)
            return y + z

        def pattern(a):
            b = torch.sigmoid(a)
            c = torch.relu(a)
            return b, c

        def replacement(x):
            return torch.exp(x), torch.abs(x)

        def comparison(x):
            y = torch.exp(x)
            z = torch.abs(x)
            return y + z

        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_replace_with_duplicated_outputs(self):
        def f(x1, x2):
            x = x1 - x2
            y = torch.sigmoid(x)
            z = torch.relu(x)
            return y + z

        def pattern(a1, a2):
            a = a1 - a2
            b = torch.sigmoid(a)
            c = torch.relu(a)
            return b, c, a

        def replacement(x1, x2):
            y1 = torch.exp(x1)
            y2 = torch.abs(x2)
            return y2, y2, y1

        def comparison(x1, x2):
            y2 = torch.abs(x2)
            return y2 + y2

        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        x1 = torch.randn(3, 4)
        x2 = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x1, x2)
        test_outs = traced.forward(x1, x2)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_with_unused_args(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y

        def pattern(x, y):
            return x + y

        def replacement(x, y):
            return x - y

        def comparison(x1, x2, x3):
            return x1 - x2

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x1 = torch.randn(3, 4)
        x2 = torch.randn(3, 4)
        x3 = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()
        placeholder_nodes = [n for n in traced.graph.nodes if n.op == "placeholder"]
        if len(placeholder_nodes) != 3:
            raise AssertionError(
                f"Expected 3 placeholder nodes, got {len(placeholder_nodes)}"
            )

        ref_outs = comparison_fn(x1, x2, x3)
        test_outs = traced.forward(x1, x2, x3)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_with_unused_results(self):
        class M(torch.nn.Module):
            def forward(self, x, y, cache):
                m = torch.mul(x, y)
                n = cache.index_copy(0, torch.tensor([0]), m)
                p = torch.ops.aten.copy.default(cache, n)
                q = torch.ops.aten.copy_.default(cache, p)  # noqa: F841
                u = torch.relu(cache)
                # check the result to ensure cache is updated before relu op
                return u

        def pattern(self_tensor, src_tensor):
            p = torch.ops.aten.copy.default(self_tensor, src_tensor)
            q = torch.ops.aten.copy_.default(self_tensor, p)
            return q

        def replacement(self_tensor, src_tensor):
            q = torch.ops.aten.copy_.default(self_tensor, src_tensor)
            return q

        def comparison(x, y, cache):
            m = torch.mul(x, y)
            n = cache.index_copy(0, torch.tensor([0]), m)
            q = torch.ops.aten.copy_.default(cache, n)  # noqa: F841
            u = torch.relu(cache)
            return u

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        x = torch.randn(1, 8)
        y = torch.randn(1, 8)
        cache = torch.randn(2, 8)
        x_clone = x.clone()
        y_clone = y.clone()
        cache_clone = cache.clone()

        ref_outs = comparison_fn(x, y, cache)
        test_outs = traced.forward(x_clone, y_clone, cache_clone)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_call_method(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x = x.dequantize()
                x = x.sigmoid()
                x = x.to(torch.float16)
                return x

        def pattern(x):
            x = x.dequantize()
            x = x.sigmoid()
            x = x.to(torch.float16)
            return x

        def replacement(x):
            return x

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(replacement)

        x1 = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x1)
        test_outs = traced.forward(x1)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_nodes_with_kwargs(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w0 = torch.nn.Parameter(torch.empty([128, 128]))
                self.b0 = torch.nn.Parameter(torch.empty([128]))

            def forward(self, in0):
                lin_res = torch.nn.functional.linear(in0, self.w0, bias=self.b0)
                mul_res = in0 * lin_res
                sum_res = mul_res + in0
                return sum_res

        def pattern(a, b, bias):
            lin_res = torch.nn.functional.linear(a, b, bias=bias)
            mul_res = a * lin_res
            return lin_res, mul_res

        def replacement(a, b, bias):
            lin_res, mul_res = wrapped_gemm_bias_mul(a, b, bias)
            return lin_res, mul_res

        traced = symbolic_trace(M())
        self._replace_once(traced, pattern, replacement)

        found_repalcement_node = False
        for node in traced.graph.nodes:
            if node.target is wrapped_gemm_bias_mul:
                found_repalcement_node = True
                break

        self.assertTrue(found_repalcement_node)

    def test_subgraph_rewriter_local_revert(self):
        # Following model will have 3 anchors as the matching candidate with the given pattern
        # Anchor 1 and 3 is a real match, but anchor 2 is not.
        # The subgraph rewriter should be able to revert the changes made while matching anchor 2.
        # Final match with anchor 3 should be successful.

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w0 = torch.nn.Parameter(torch.empty([128, 128]))
                self.b0 = torch.nn.Parameter(torch.empty([128]))
                self.w1 = torch.nn.Parameter(torch.empty([128, 128]))
                self.b1 = torch.nn.Parameter(torch.empty([128]))
                self.w2 = torch.nn.Parameter(torch.empty([128, 128]))
                self.b2 = torch.nn.Parameter(torch.empty([128]))
                self.w3 = torch.nn.Parameter(torch.empty([128, 128]))
                self.b3 = torch.nn.Parameter(torch.empty([128]))
                self.w4 = torch.nn.Parameter(torch.empty([128, 128]))
                self.b4 = torch.nn.Parameter(torch.empty([128]))

            def forward(self, in0, in1):
                lin_res_1 = torch.nn.functional.linear(in1, self.w0, bias=self.b0)
                lin_res_2 = torch.nn.functional.linear(lin_res_1, self.w1, bias=self.b1)
                # potential match at anchor 1
                mul_res_1 = in1 * lin_res_2
                sum_res_1 = mul_res_1 + in1
                lin_res_3 = torch.nn.functional.linear(sum_res_1, self.w2, bias=self.b2)
                sigmoid_res_1 = torch.sigmoid(lin_res_3)
                # potential match at anchor 2
                mul_res_2 = lin_res_3 * sigmoid_res_1
                lin_res_4 = torch.nn.functional.linear(in0, self.w3, bias=self.b3)
                lin_res_5 = torch.nn.functional.linear(lin_res_4, self.w4, bias=self.b4)
                # potential match at anchor 3
                mul_res_3 = in0 * lin_res_5
                sum_res_2 = mul_res_3 + in0
                cat_res = torch.cat(
                    [mul_res_2, sum_res_2],
                    dim=1,
                )
                return cat_res

        def gemm_bias_mul_pattern_with_c(a, b, bias, c):
            lin_res = torch.nn.functional.linear(a, b, bias=bias)
            mul_res = c * lin_res
            return lin_res, mul_res

        def gemm_bias_mul_replacement_with_c(a, b, bias, c):
            lin_res, mul_res = wrapped_gemm_bias_mul_with_c(a, b, bias, c)
            return lin_res, mul_res

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern(
            traced, gemm_bias_mul_pattern_with_c, gemm_bias_mul_replacement_with_c
        )

        self.assertEqual(len(matches), 2)

        repalcement_node_found = 0
        for node in traced.graph.nodes:
            if node.target is wrapped_gemm_bias_mul_with_c:
                repalcement_node_found += 1

        self.assertEqual(repalcement_node_found, 2)

    def test_replace_pattern_with_filters(self):
        class M(torch.nn.Module):
            def forward(self, x, scale, zero_point):
                # Match, second input to add is a scalar
                x = x.dequantize()
                x = torch.add(x, 2)
                x = x.relu()
                x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

                y = x + 1
                # NOT a match, second input to add is NOT a scalar
                x = x.dequantize()
                x = torch.add(x, y)
                x = x.relu()
                x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

                return x

        def BinaryOpScalarReLUPattern(x, num, scale, zero_point):
            x = x.dequantize()
            x = torch.add(x, num)
            x = x.relu()
            x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
            return x

        def BinaryOpScalarReLUReplacement(x, num, scale, zero_point):
            x = torch.mul(x, num)
            return x

        def second_input_is_scalar(match, original_graph, pattern_graph):
            """check the node that's matched to the second input of the pattern graph
            is a scalar number
            """
            input_idx = 0
            for node in pattern_graph.nodes:
                if node.op == "placeholder":
                    if input_idx == 1:
                        num_node = node
                    input_idx += 1
            return isinstance(match.nodes_map[num_node], (int, float))

        def check_replacement_nodes(self, traced, matches):
            replacement_nodes_in_graph = [
                node for node in traced.graph.nodes if node.target == torch.mul
            ]
            replacement_nodes_in_res = [r for m in matches for r in m.replacements]
            self.assertEqual(
                len(replacement_nodes_in_graph), len(replacement_nodes_in_res)
            )
            self.assertEqual(replacement_nodes_in_graph, replacement_nodes_in_res)
            return len(replacement_nodes_in_graph)

        # match without filter, should find 2 match
        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced, BinaryOpScalarReLUPattern, BinaryOpScalarReLUReplacement, None
        )
        self.assertEqual(len(matches), 2)
        self.assertEqual(check_replacement_nodes(self, traced, matches), 2)

        # match with filter, should find 1 match
        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced,
            BinaryOpScalarReLUPattern,
            BinaryOpScalarReLUReplacement,
            [second_input_is_scalar],
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(check_replacement_nodes(self, traced, matches), 1)

    def test_replace_pattern_populates_replacement_output_meta_val(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.div.Tensor(x, y)

        def pattern(x, y):
            return torch.ops.aten.div.Tensor(x, y)

        def replacement(x, y):
            return torch.ops.aten.mul.Tensor(x, y)

        x, y = torch.randn(4), torch.randn(4)
        ep = torch.export.export(M(), (x, y))
        gm = ep.graph_module

        self._replace_once(gm, pattern, replacement)

        replacement_nodes = [
            node for node in gm.graph.nodes if node.target == torch.ops.aten.mul.Tensor
        ]
        self.assertEqual(len(replacement_nodes), 1)
        replacement_node = replacement_nodes[0]
        val = replacement_node.meta.get("val")
        self.assertIsInstance(val, torch.Tensor)
        self.assertEqual(val.shape, torch.Size([4]))
        self.assertEqual(val.dtype, torch.float32)

    def test_replace_pattern_populates_shape_changing_replacement_meta_val(self):
        def replacement(x):
            return torch.ops.aten.reshape.default(x, [6])

        ep = torch.export.export(_ReluModel(), (torch.randn(2, 3),))
        gm = ep.graph_module

        self._replace_once(gm, _relu_pattern, replacement)

        replacement_node = self._node_with_target(gm, torch.ops.aten.reshape.default)
        val = replacement_node.meta.get("val")
        self.assertIsInstance(val, torch.Tensor)
        self.assertEqual(val.shape, torch.Size([6]))
        input_val = next(
            node.meta["val"] for node in gm.graph.nodes if node.op == "placeholder"
        )
        self.assertEqual(
            val.untyped_storage()._cdata, input_val.untyped_storage()._cdata
        )

    def test_replace_pattern_populates_common_replacement_meta_val(self):
        def operator_replacement(x):
            return x * x

        def torch_mul_replacement(x):
            return torch.mul(x, x)

        def packet_replacement(x):
            return torch.ops.aten.mul(x, x)

        def prims_replacement(x):
            return torch.ops.prims.mul.default(x, x)

        def prims_packet_replacement(x):
            return torch.ops.prims.mul(x, x)

        def method_relu_replacement(x):
            return x.relu()

        def method_add_replacement(x):
            return x.add(x)

        def method_reshape_replacement(x):
            return x.reshape(3, 2)

        def method_view_replacement(x):
            return x.view(3, 2)

        def torch_relu_replacement(x):
            return torch.relu(x)

        def torch_cat_replacement(x):
            return torch.cat((x, x), dim=0)

        def torch_reshape_replacement(x):
            return torch.reshape(x, (3, 2))

        def torch_stack_replacement(x):
            return torch.stack((x, x), dim=0)

        def torch_sin_replacement(x):
            return torch.sin(x)

        for replacement, target, shape in [
            (operator_replacement, operator.mul, torch.Size([2, 3])),
            (torch_mul_replacement, torch.mul, torch.Size([2, 3])),
            (packet_replacement, torch.ops.aten.mul, torch.Size([2, 3])),
            (prims_replacement, torch.ops.prims.mul.default, torch.Size([2, 3])),
            (prims_packet_replacement, torch.ops.prims.mul, torch.Size([2, 3])),
            (method_relu_replacement, "relu", torch.Size([2, 3])),
            (method_add_replacement, "add", torch.Size([2, 3])),
            (method_reshape_replacement, "reshape", torch.Size([3, 2])),
            (method_view_replacement, "view", torch.Size([3, 2])),
            (torch_relu_replacement, torch.relu, torch.Size([2, 3])),
            (torch_cat_replacement, torch.cat, torch.Size([4, 3])),
            (torch_reshape_replacement, torch.reshape, torch.Size([3, 2])),
            (torch_stack_replacement, torch.stack, torch.Size([2, 2, 3])),
            (torch_sin_replacement, torch.sin, torch.Size([2, 3])),
        ]:
            with self.subTest(replacement=replacement.__name__):
                gm = torch.export.export(_NegModel(), (torch.randn(2, 3),)).graph_module

                self._replace_once(gm, _neg_pattern, replacement)

                replacement_node = self._node_with_target(gm, target)
                val = replacement_node.meta.get("val")
                self.assertIsInstance(val, torch.Tensor)
                self.assertEqual(val.shape, shape)

    def test_replace_pattern_populates_getitem_replacement_meta_val(self):
        def max_replacement(x):
            return torch.ops.aten.max.dim(x, 1)[0]

        def slice_replacement(x):
            return x[:, 0]

        for replacement in (max_replacement, slice_replacement):
            with self.subTest(replacement=replacement.__name__):
                gm = self._export_relu()
                self._replace_once(gm, _relu_pattern, replacement)

                getitem_node = self._node_with_target(gm, operator.getitem)
                val = getitem_node.meta.get("val")
                self.assertIsInstance(val, torch.Tensor)
                self.assertEqual(val.shape, torch.Size([2]))

    def test_replace_pattern_populates_get_attr_replacement_meta_val(self):
        class Replacement(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("offset", torch.ones(2, 3))

            def forward(self, x):
                return torch.ops.aten.add.Tensor(x, self.offset)

        def numel_filter(match, original_graph, pattern_graph):
            val = match.returning_nodes[0].meta.get("val")
            return isinstance(val, torch.Tensor) and val.numel() == 6

        ep = torch.export.export(_ReluModel(), (torch.randn(2, 3),))
        gm = ep.graph_module

        self._replace_once(gm, _relu_pattern, Replacement())

        offset_node = next(node for node in gm.graph.nodes if node.target == "offset")
        self.assertIsInstance(offset_node.meta.get("val"), torch.Tensor)
        add_node = self._node_with_target(gm, torch.ops.aten.add.Tensor)
        self.assertIsInstance(add_node.meta.get("val"), torch.Tensor)

        def add_pattern(x, y):
            return torch.ops.aten.add.Tensor(x, y)

        def mul_replacement(x, y):
            return torch.ops.aten.mul.Tensor(x, y)

        matches = subgraph_rewriter.replace_pattern_with_filters(
            gm, add_pattern, mul_replacement, [numel_filter]
        )
        self.assertEqual(len(matches), 1)

    def test_replace_pattern_meta_copy_prefers_destination_get_attr(self):
        class Replacement(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("weight", torch.full((3, 4), 7.0))

            def forward(self, x):
                return torch.ops.aten.mm.default(x, self.weight)

        gm = symbolic_trace(_ReluModel())
        gm.register_buffer("weight", torch.full((3, 5), 2.0))
        original_weight = gm.weight
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        placeholder.meta["val"] = torch.ones(2, 3)

        replacement_gm = symbolic_trace(Replacement())
        replacement_weight_node = self._node_with_target(replacement_gm, "weight")
        replacement_mm_node = self._node_with_target(
            replacement_gm, torch.ops.aten.mm.default
        )
        with FakeTensorMode() as mode:
            replacement_weight_node.meta["val"] = mode.from_tensor(
                replacement_gm.weight
            )
            replacement_mm_node.meta["val"] = mode.from_tensor(torch.empty(2, 4))

        self._replace_once(gm, _relu_pattern, replacement_gm)
        self.assertIs(gm.weight, original_weight)

        weight_node = next(node for node in gm.graph.nodes if node.target == "weight")
        self.assertEqual(weight_node.meta["val"].shape, torch.Size([3, 5]))
        mm_node = self._node_with_target(gm, torch.ops.aten.mm.default)
        self.assertEqual(mm_node.meta["val"].shape, torch.Size([2, 5]))
        self.assertEqual(gm(torch.ones(2, 3)).shape, torch.Size([2, 5]))

    def test_replace_pattern_meta_copy_recomputes_rebound_get_attr(self):
        replacement = torch.nn.Module()
        replacement.count = 2
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        count = graph.get_attr("count")
        add = graph.call_function(operator.add, (x, count))
        graph.output(add)
        replacement_gm = torch.fx.GraphModule(replacement, graph)
        count.meta["val"] = 2
        replacement_gm.count = 3

        gm = symbolic_trace(_ReluModel())
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        placeholder.meta["val"] = torch.ones(2)
        self._replace_once(gm, _relu_pattern, replacement_gm)
        self.assertEqual(self._node_with_target(gm, "count").meta["val"], 3)
        self.assertEqual(gm(torch.ones(2)), torch.full((2,), 4.0))

    def test_replace_pattern_meta_copy_invalidates_raw_graph_get_attr(self):
        gm = symbolic_trace(_ReluModel())
        gm.register_buffer("weight", torch.full((3, 5), 2.0))
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        placeholder.meta["val"] = torch.ones(2, 3)

        replacement_graph = torch.fx.Graph()
        x = replacement_graph.placeholder("x")
        weight = replacement_graph.get_attr("weight")
        mm = replacement_graph.call_function(torch.ops.aten.mm.default, (x, weight))
        replacement_graph.output(mm)
        with FakeTensorMode() as mode:
            weight.meta["val"] = mode.from_tensor(torch.empty(3, 4))
            mm.meta["val"] = mode.from_tensor(torch.empty(2, 4))

        self._replace_once(gm, _relu_pattern, replacement_graph)

        weight_node = next(node for node in gm.graph.nodes if node.target == "weight")
        self.assertEqual(weight_node.meta["val"].shape, torch.Size([3, 5]))
        mm_node = self._node_with_target(gm, torch.ops.aten.mm.default)
        self.assertEqual(mm_node.meta["val"].shape, torch.Size([2, 5]))
        self.assertEqual(gm(torch.ones(2, 3)).shape, torch.Size([2, 5]))

    def test_replace_pattern_meta_copy_does_not_read_destination_descriptor(self):
        descriptor_accesses = 0

        gm = symbolic_trace(_ReluModel())
        gm_type = type(gm)

        def read_danger(self):
            nonlocal descriptor_accesses
            descriptor_accesses += 1
            return torch.ones(2, 3)

        gm_type.danger = property(read_danger)
        self.addCleanup(delattr, gm_type, "danger")
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        placeholder.meta["val"] = torch.ones(2, 3)

        replacement_graph = torch.fx.Graph()
        replacement_graph.placeholder("x")
        danger = replacement_graph.get_attr("danger")
        replacement_graph.output(danger)

        self._replace_once(gm, _relu_pattern, replacement_graph)
        self.assertEqual(descriptor_accesses, 0)
        danger_node = next(node for node in gm.graph.nodes if node.target == "danger")
        self.assertNotIn("val", danger_node.meta)

    def test_replace_pattern_meta_copy_invalidates_destination_call_module(self):
        class Replacement(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, x):
                return torch.sin(self.linear(x))

        gm = symbolic_trace(_ReluModel())
        destination_linear = torch.nn.Linear(3, 5)
        gm.add_module("linear", destination_linear)
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        placeholder.meta["val"] = torch.ones(2, 3)

        replacement_gm = symbolic_trace(Replacement())
        with FakeTensorMode() as mode:
            for node in replacement_gm.graph.nodes:
                if node.target == "linear":
                    node.meta["val"] = mode.from_tensor(torch.empty(2, 4))
                elif node.target == torch.sin:
                    node.meta["val"] = mode.from_tensor(torch.empty(2, 4))

        self._replace_once(gm, _relu_pattern, replacement_gm)
        self.assertIs(gm.linear, destination_linear)

        linear_node = next(node for node in gm.graph.nodes if node.target == "linear")
        sin_node = next(node for node in gm.graph.nodes if node.target == torch.sin)
        self.assertNotIn("val", linear_node.meta)
        self.assertNotIn("val", sin_node.meta)
        self.assertEqual(gm(torch.ones(2, 3)).shape, torch.Size([2, 5]))

    def test_replace_pattern_meta_copy_invalidates_call_module_inputs(self):
        class Consumer(torch.nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten.mm.default(x, weight)

        consumer = Consumer()
        gm = symbolic_trace(_ReluModel())
        gm.register_buffer("weight", torch.full((3, 5), 2.0))
        gm.add_module("consumer", consumer)
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        placeholder.meta["val"] = torch.ones(2, 3)

        replacement_root = torch.nn.Module()
        replacement_root.register_buffer("weight", torch.full((3, 4), 7.0))
        replacement_root.add_module("consumer", consumer)
        replacement_graph = torch.fx.Graph()
        x = replacement_graph.placeholder("x")
        weight = replacement_graph.get_attr("weight")
        call_module = replacement_graph.call_module("consumer", (x, weight))
        sin = replacement_graph.call_function(torch.sin, (call_module,))
        replacement_graph.output(sin)
        replacement_gm = torch.fx.GraphModule(replacement_root, replacement_graph)
        self.assertIs(gm.consumer, replacement_gm.consumer)

        with FakeTensorMode() as mode:
            weight.meta["val"] = mode.from_tensor(torch.empty(3, 4))
            call_module.meta["val"] = mode.from_tensor(torch.empty(2, 4))
            sin.meta["val"] = mode.from_tensor(torch.empty(2, 4))

        self._replace_once(gm, _relu_pattern, replacement_gm)

        weight_node = next(node for node in gm.graph.nodes if node.target == "weight")
        call_module_node = self._node_with_target(gm, "consumer")
        sin_node = next(node for node in gm.graph.nodes if node.target == torch.sin)
        self.assertEqual(weight_node.meta["val"].shape, torch.Size([3, 5]))
        self.assertNotIn("val", call_module_node.meta)
        self.assertNotIn("val", sin_node.meta)
        self.assertEqual(gm(torch.ones(2, 3)).shape, torch.Size([2, 5]))

    def test_replace_pattern_populates_parameter_meta_val(self):
        class Replacement(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(3, 4))

            def forward(self, x):
                return torch.ops.aten.mm.default(x, self.weight)

        gm = self._export_relu()

        self._replace_once(gm, _relu_pattern, Replacement())

        weight_node = next(node for node in gm.graph.nodes if node.target == "weight")
        self.assertIsInstance(weight_node.meta.get("val"), FakeTensor)
        mm_node = self._node_with_target(gm, torch.ops.aten.mm.default)
        mm_val = mm_node.meta.get("val")
        self.assertIsInstance(mm_val, FakeTensor)
        self.assertEqual(mm_val.shape, torch.Size([2, 4]))

    def test_replace_pattern_populates_real_tensor_meta_val(self):
        def replacement(x, y):
            return torch.ops.aten.mul.Tensor(x, y)

        x, y = torch.randn(2, 3), torch.randn(2, 3)
        gm = symbolic_trace(_AddModel())
        placeholder_nodes = [
            node for node in gm.graph.nodes if node.op == "placeholder"
        ]
        self.assertEqual(len(placeholder_nodes), 2)
        placeholder_nodes[0].meta["val"] = x
        placeholder_nodes[1].meta["val"] = y

        self._replace_once(gm, _add_pattern, replacement)

        replacement_node = self._node_with_target(gm, torch.ops.aten.mul.Tensor)
        val = replacement_node.meta.get("val")

        self.assertIsInstance(val, FakeTensor)
        self.assertEqual(val.shape, torch.Size([2, 3]))
        self.assertEqual(val.dtype, torch.float32)

    def test_replace_pattern_populates_symbolic_scalar_meta_val(self):
        def replacement(x):
            size = torch.ops.aten.sym_size.int(x, 0)
            return torch.ops.aten.view.default(x, [size, -1])

        ep = torch.export.export(
            _ReluModel(),
            (torch.randn(3, 4),),
            dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        )
        gm = ep.graph_module

        self._replace_once(gm, _relu_pattern, replacement)

        sym_size_node = self._node_with_target(gm, torch.ops.aten.sym_size.int)
        self.assertIs(type(sym_size_node.meta.get("val")), torch.SymInt)
        view_node = self._node_with_target(gm, torch.ops.aten.view.default)
        view_val = view_node.meta.get("val")
        self.assertIsInstance(view_val, FakeTensor)
        self.assertIs(type(view_val.shape[0]), torch.SymInt)
        self.assertEqual(view_val.shape[1], 4)

        shape_env = ShapeEnv()
        for value, value_type in [
            (shape_env.create_unbacked_symint(), torch.SymInt),
            (shape_env.create_unbacked_symfloat(), torch.SymFloat),
            (shape_env.create_unbacked_symbool(), torch.SymBool),
        ]:
            with self.subTest(value_type=value_type):
                self.assertIs(type(value), value_type)
                self.assertIs(subgraph_rewriter._copy_meta_val(value), value)

    def test_replace_pattern_meta_copy_does_not_run_custom_meta_value(self):
        custom_meta_call_count = 0

        class CustomMeta:
            def __mul__(self, other):
                nonlocal custom_meta_call_count
                custom_meta_call_count += 1
                return self

            def __rmul__(self, other):
                nonlocal custom_meta_call_count
                custom_meta_call_count += 1
                return self

        def replacement(x, y):
            return x * y

        ep = torch.export.export(_AddModel(), (torch.randn(4), torch.randn(4)))
        gm = ep.graph_module
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["val"] = CustomMeta()

        self._replace_once(gm, _add_pattern, replacement)
        self.assertEqual(custom_meta_call_count, 0)
        replacement_node = self._node_with_target(gm, operator.mul)
        self.assertNotIn("val", replacement_node.meta)

    def test_replace_pattern_meta_copy_skips_cyclic_metadata(self):
        def replacement(x):
            return torch.ops.aten.neg.default(x)

        cyclic_meta = [torch.randn(2, 3)]
        cyclic_meta.append(cyclic_meta)
        self.assertFalse(subgraph_rewriter._is_safe_meta_value(cyclic_meta))
        self.assertFalse(subgraph_rewriter._contains_tensor(cyclic_meta))

        gm = symbolic_trace(_ReluModel())
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        placeholder.meta["val"] = cyclic_meta

        self._replace_once(gm, _relu_pattern, replacement)
        replacement_node = self._node_with_target(gm, torch.ops.aten.neg.default)
        self.assertNotIn("val", replacement_node.meta)

    @skipIfTorchDynamo("TorchDynamo inspects the custom metaclass")
    def test_replace_pattern_meta_copy_does_not_inspect_custom_metaclass(self):
        metaclass_accesses = 0

        class CustomMeta(type):
            def __getattribute__(cls, name):
                nonlocal metaclass_accesses
                metaclass_accesses += 1
                return super().__getattribute__(name)

            def __eq__(cls, other):
                nonlocal metaclass_accesses
                metaclass_accesses += 1
                return super().__eq__(other)

            def __hash__(cls):
                nonlocal metaclass_accesses
                metaclass_accesses += 1
                return super().__hash__()

        class CustomValue(metaclass=CustomMeta):
            pass

        value = CustomValue()
        metaclass_accesses = 0
        self.assertFalse(subgraph_rewriter._is_safe_meta_value(value))
        self.assertFalse(subgraph_rewriter._contains_tensor(value))
        self.assertEqual(metaclass_accesses, 0)

    def test_replace_pattern_meta_copy_does_not_inspect_custom_target(self):
        target_accesses = 0

        class CustomTarget:
            @property
            def __name__(self):
                nonlocal target_accesses
                target_accesses += 1
                return "sin"

            def __hash__(self):
                nonlocal target_accesses
                target_accesses += 1
                return id(self)

            def __call__(self, x):
                return x

        class CustomMethod(str):
            __slots__ = ()

            def __hash__(self):
                nonlocal target_accesses
                target_accesses += 1
                return super().__hash__()

        self.assertFalse(
            subgraph_rewriter._is_safe_meta_propagation_target(CustomTarget(), True)
        )
        method_target = CustomMethod("relu")
        self.assertFalse(
            type(method_target) is str
            and subgraph_rewriter._is_safe_meta_propagation_method_name(method_target)
        )
        self.assertEqual(target_accesses, 0)

    def test_replace_pattern_meta_copy_does_not_run_tensor_subclass(self):
        subclass_call_count = 0

        class CustomTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                nonlocal subclass_call_count
                subclass_call_count += 1
                return super().__torch_function__(func, types, args, kwargs)

        def replacement(x, y):
            return x * y

        ep = torch.export.export(_AddModel(), (torch.randn(4), torch.randn(4)))
        gm = ep.graph_module
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["val"] = CustomTensor(torch.randn(4))
        subclass_call_count = 0

        self._replace_once(gm, _add_pattern, replacement)
        self.assertEqual(subclass_call_count, 0)
        replacement_node = self._node_with_target(gm, operator.mul)
        self.assertNotIn("val", replacement_node.meta)

    def test_replace_pattern_meta_copy_does_not_run_tuple_subclass(self):
        constructor_call_count = 0

        class max(tuple):
            __module__ = "torch.return_types"
            __slots__ = ()
            _fields = ("values", "indices")

            def __new__(cls, values, indices):
                nonlocal constructor_call_count
                constructor_call_count += 1
                return tuple.__new__(cls, (values, indices))

        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        def pattern(x):
            return x + 1

        def replacement(x):
            return x[0]

        gm = symbolic_trace(M())
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        placeholder.meta["val"] = max(torch.randn(4), torch.zeros(4, dtype=torch.long))
        constructor_call_count = 0

        self._replace_once(gm, pattern, replacement)
        self.assertEqual(constructor_call_count, 0)
        replacement_node = self._node_with_target(gm, operator.getitem)
        self.assertNotIn("val", replacement_node.meta)

    def test_replace_pattern_meta_copy_skips_unsafe_call_method(self):
        replacement_graph = torch.fx.Graph()
        x = replacement_graph.placeholder("x")
        method = replacement_graph.call_method("__getattribute__", (x, "relu"))
        replacement_graph.output(method)

        ep = torch.export.export(_NegModel(), (torch.randn(2, 3),))
        gm = ep.graph_module

        self._replace_once(gm, _neg_pattern, replacement_graph)
        replacement_node = self._node_with_target(gm, "__getattribute__")
        self.assertNotIn("val", replacement_node.meta)

    def test_replace_pattern_meta_copy_skips_non_tensor_method(self):
        replacement_graph = torch.fx.Graph()
        x = replacement_graph.placeholder("x")
        method = replacement_graph.call_method("sym_size", (x, 0))
        replacement_graph.output(method)

        gm = symbolic_trace(_NegModel())
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        placeholder.meta["val"] = torch.randn(2, 3)

        self._replace_once(gm, _neg_pattern, replacement_graph)
        replacement_node = self._node_with_target(gm, "sym_size")
        self.assertNotIn("val", replacement_node.meta)

    def test_replace_pattern_preserves_preannotated_replacement_meta_val(self):
        def replacement(x):
            return torch.ops.aten.neg.default(x)

        replacement_gm = symbolic_trace(replacement)
        replacement_node = self._node_with_target(
            replacement_gm, torch.ops.aten.neg.default
        )
        with FakeTensorMode() as mode:
            replacement_meta_val = mode.from_tensor(torch.empty(2, 3))
        replacement_node.meta["val"] = replacement_meta_val

        ep = torch.export.export(_ReluModel(), (torch.randn(4, 5),))
        gm = ep.graph_module

        self._replace_once(gm, _relu_pattern, replacement_gm)
        copied_node = self._node_with_target(gm, torch.ops.aten.neg.default)
        self.assertIs(copied_node.meta.get("val"), replacement_meta_val)

    def test_replace_pattern_replaces_none_replacement_meta_val(self):
        def replacement(x):
            return torch.ops.aten.reshape.default(x, [6])

        replacement_gm = symbolic_trace(replacement)
        replacement_node = self._node_with_target(
            replacement_gm, torch.ops.aten.reshape.default
        )
        replacement_node.meta["val"] = None

        ep = torch.export.export(_ReluModel(), (torch.randn(2, 3),))
        gm = ep.graph_module

        self._replace_once(gm, _relu_pattern, replacement_gm)
        copied_node = self._node_with_target(gm, torch.ops.aten.reshape.default)
        val = copied_node.meta.get("val")
        self.assertIsInstance(val, torch.Tensor)
        self.assertEqual(val.shape, torch.Size([6]))

    def test_replace_pattern_with_filters_populates_returning_node_meta_val(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.div.Tensor(x, y)

        def div_pattern(x, y):
            return torch.ops.aten.div.Tensor(x, y)

        def mul_replacement(x, y):
            return torch.ops.aten.mul.Tensor(x, y)

        def numel_filter(match, original_graph, pattern_graph):
            val = match.returning_nodes[0].meta.get("val")
            return isinstance(val, torch.Tensor) and val.numel() == 4

        x, y = torch.randn(4), torch.randn(4)
        ep = torch.export.export(M(), (x, y))
        gm = ep.graph_module

        matches = subgraph_rewriter.replace_pattern_with_filters(
            gm, div_pattern, mul_replacement, [numel_filter]
        )
        self.assertEqual(len(matches), 1)
        mul_node = matches[0].replacements[0]
        self.assertEqual(mul_node.target, torch.ops.aten.mul.Tensor)
        self.assertIsInstance(mul_node.meta.get("val"), torch.Tensor)

        def mul_pattern(x, y):
            return torch.ops.aten.mul.Tensor(x, y)

        def add_replacement(x, y):
            return torch.ops.aten.add.Tensor(x, y)

        matches = subgraph_rewriter.replace_pattern_with_filters(
            gm, mul_pattern, add_replacement, [numel_filter]
        )
        self.assertEqual(len(matches), 1)
        add_node = matches[0].replacements[0]
        self.assertEqual(add_node.target, torch.ops.aten.add.Tensor)
        self.assertIsInstance(add_node.meta.get("val"), torch.Tensor)

    def test_replace_pattern_meta_copy_does_not_run_wrapped_replacement(self):
        self._reset_side_effect_replacement_call_count()
        self.addCleanup(self._reset_side_effect_replacement_call_count)

        def replacement(x):
            return side_effect_replacement(x)

        ep = torch.export.export(_ReluModel(), (torch.randn(2, 3),))
        gm = ep.graph_module

        self._replace_once(gm, _relu_pattern, replacement)
        self.assertEqual(_side_effect_replacement_call_count, 0)
        replacement_nodes = [
            node for node in gm.graph.nodes if node.target == side_effect_replacement
        ]
        self.assertEqual(len(replacement_nodes), 1)
        replacement_node = replacement_nodes[0]
        self.assertEqual(replacement_node.target, side_effect_replacement)
        self.assertNotIn("val", replacement_node.meta)

    @skipIfTorchDynamo("TorchDynamo manages its own dispatch modes")
    def test_replace_pattern_meta_copy_disables_ambient_dispatch_modes(self):
        calls = []

        class RecordingMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                calls.append(func)
                return func(*args, **(kwargs or {}))

        def replacement(x):
            value = torch.ops.aten.neg.default(x)
            return torch.ops.prims.mul.default(value, x).relu()

        gm = self._export_relu()
        sentinel = torch.ones(1)
        with RecordingMode():
            matches = subgraph_rewriter.replace_pattern(gm, _relu_pattern, replacement)
            self.assertEqual(calls, [])
            torch.ops.aten.neg.default(sentinel)

        self.assertEqual(len(matches), 1)
        self.assertEqual(calls, [torch.ops.aten.neg.default])
        for target in (torch.ops.aten.neg.default, torch.ops.prims.mul.default, "relu"):
            node = next(node for node in gm.graph.nodes if node.target == target)
            self.assertIsInstance(node.meta.get("val"), FakeTensor)

        gm = self._export_relu()
        with FakeTensorMode():
            self._replace_once(gm, _relu_pattern, replacement)
        node = self._node_with_target(gm, torch.ops.aten.neg.default)
        self.assertIsInstance(node.meta.get("val"), FakeTensor)

    def test_replace_pattern_preserves_unsupported_preannotated_meta_val(self):
        replacement_graph = torch.fx.Graph()
        x = replacement_graph.placeholder("x")
        replacement = replacement_graph.call_function(torch.ops.aten.neg.default, (x,))
        replacement_graph.output(replacement)
        replacement_meta_val = object()
        replacement.meta["val"] = replacement_meta_val

        gm = torch.export.export(_ReluModel(), (torch.randn(4, 5),)).graph_module
        self._replace_once(gm, _relu_pattern, replacement_graph)
        replacement_node = self._node_with_target(gm, torch.ops.aten.neg.default)
        self.assertIs(replacement_node.meta["val"], replacement_meta_val)

    def test_replace_pattern_meta_copy_does_not_run_custom_op_fake(self):
        fake_call_count = 0

        with torch.library._scoped_library("subgraph_rewriter_test", "FRAGMENT") as lib:
            torch.library.define(
                "subgraph_rewriter_test::custom_replacement",
                "(Tensor x) -> Tensor",
                lib=lib,
            )

            @torch.library.impl(
                "subgraph_rewriter_test::custom_replacement", "CPU", lib=lib
            )
            def custom_replacement_cpu(x):
                return x.clone()

            @torch.library.register_fake(
                "subgraph_rewriter_test::custom_replacement", lib=lib
            )
            def custom_replacement_fake(x):
                nonlocal fake_call_count
                fake_call_count += 1
                return x.clone()

            def replacement(x):
                return torch.ops.subgraph_rewriter_test.custom_replacement.default(x)

            ep = torch.export.export(_ReluModel(), (torch.randn(2, 3),))
            gm = ep.graph_module

            self._replace_once(gm, _relu_pattern, replacement)
            self.assertEqual(fake_call_count, 0)
            replacement_node = self._node_with_target(
                gm, torch.ops.subgraph_rewriter_test.custom_replacement.default
            )
            self.assertNotIn("val", replacement_node.meta)

    def test_replace_pattern_meta_copy_does_not_run_python_defined_builtin_fake(self):
        fake_call_count = 0

        for namespace in ("aten", "prims"):
            qualname = f"{namespace}::_subgraph_rewriter_test_custom"
            with torch.library._scoped_library(namespace, "FRAGMENT") as lib:
                torch.library.define(qualname, "(Tensor x) -> Tensor", lib=lib)

                @torch.library.register_fake(qualname, lib=lib)
                def custom_replacement_fake(x):
                    nonlocal fake_call_count
                    fake_call_count += 1
                    return x.clone()

                ops = {"aten": torch.ops.aten, "prims": torch.ops.prims}[namespace]
                packet = ops._subgraph_rewriter_test_custom

                def overload_replacement(x):
                    return packet.default(x)

                def packet_replacement(x):
                    return packet(x)

                for replacement, target in [
                    (overload_replacement, packet.default),
                    (packet_replacement, packet),
                ]:
                    with self.subTest(
                        namespace=namespace, replacement=replacement.__name__
                    ):
                        gm = self._export_relu()
                        self._replace_once(gm, _relu_pattern, replacement)
                        self.assertEqual(fake_call_count, 0)
                        replacement_node = self._node_with_target(gm, target)
                        self.assertNotIn("val", replacement_node.meta)

    def test_replace_pattern_meta_copy_skips_registered_aten_fake(self):
        def overload_replacement(x):
            return torch.ops.aten.neg.default(x)

        def method_replacement(x):
            return x.neg()

        def packet_replacement(x):
            return torch.ops.aten.mul(x, x)

        real_get = _simple_registry.get
        fake_entry = mock.Mock()
        fake_entry.fake_impl.kernel = object()
        with mock.patch.object(
            _simple_registry,
            "get",
            side_effect=lambda qualname: (
                fake_entry
                if qualname in ("aten::neg", "aten::mul.Scalar")
                else real_get(qualname)
            ),
        ):
            for replacement, target in (
                (overload_replacement, torch.ops.aten.neg.default),
                (method_replacement, "neg"),
                (packet_replacement, torch.ops.aten.mul),
            ):
                gm = self._export_relu()
                self._replace_once(gm, _relu_pattern, replacement)
                replacement_node = self._node_with_target(gm, target)
                if target == torch.ops.aten.mul:
                    self.assertIsInstance(replacement_node.meta.get("val"), FakeTensor)
                else:
                    self.assertNotIn("val", replacement_node.meta)

    def test_replace_pattern_meta_copy_skips_python_dispatch_overrides(self):
        calls = 0

        def evil(*args, **kwargs):
            nonlocal calls
            calls += 1
            return args[-1]

        gm = self._export_relu()
        with torch.library._scoped_library("aten", "IMPL") as lib:
            lib.impl("neg", evil, "Meta", allow_override=True)
            torch.library.register_torch_dispatch(
                torch.ops.aten.neg.default, FakeTensorMode, evil, lib=lib
            )
            self._replace_once(gm, _relu_pattern, lambda x: x.neg())
        self.assertEqual(calls, 0)
        self.assertNotIn("val", self._node_with_target(gm, "neg").meta)

        class DerivedFakeMode(FakeTensorMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                nonlocal calls
                if func == torch.ops.aten.sin.default:
                    calls += 1
                return super().__torch_dispatch__(func, types, args, kwargs)

        gm = symbolic_trace(_ReluModel())
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        with DerivedFakeMode() as mode:
            placeholder.meta["val"] = mode.from_tensor(torch.randn(2, 3))
        with torch.library._scoped_library("aten", "IMPL") as lib:
            torch.library.register_torch_dispatch(
                torch.ops.aten.sin.default, DerivedFakeMode, evil, lib=lib
            )
            self._replace_once(gm, _relu_pattern, lambda x: torch.sin(x))
        self.assertEqual(calls, 0)
        self.assertNotIn("val", self._node_with_target(gm, torch.sin).meta)

    def test_replace_pattern_meta_copy_skips_python_tensor_method(self):
        calls = 0

        def evil(self):
            nonlocal calls
            calls += 1
            return self

        self.assertTrue(subgraph_rewriter._is_safe_meta_propagation_method_name("neg"))
        original = torch.Tensor.neg
        torch.Tensor.neg = evil
        try:
            gm = self._export_relu()
            self._replace_once(gm, _relu_pattern, lambda x: x.neg())
        finally:
            torch.Tensor.neg = original
        self.assertEqual(calls, 0)
        self.assertNotIn("val", self._node_with_target(gm, "neg").meta)

    def test_replace_pattern_meta_copy_skips_data_dependent_op(self):
        def nonzero_replacement(x):
            return torch.ops.aten.nonzero.default(x)

        def guarded_view_replacement(x):
            return torch.ops.aten.view.default(x, [2, -1])

        def item_replacement(x):
            return torch.ops.aten.item.default(x)

        for replacement, target in (
            (nonzero_replacement, torch.ops.aten.nonzero.default),
            (guarded_view_replacement, torch.ops.aten.view.default),
            (item_replacement, torch.ops.aten.item.default),
        ):
            gm = torch.export.export(
                _ReluModel(),
                (torch.randn(4, 3),),
                dynamic_shapes={"x": {0: torch.export.Dim("batch", min=3, max=8)}},
            ).graph_module
            fake_mode = next(
                node.meta["val"].fake_mode
                for node in gm.graph.nodes
                if node.op == "placeholder"
            )
            shape_env = fake_mode.shape_env
            before = (
                list(shape_env.pending_fresh_unbacked_symbols),
                shape_env.counter["ignored_backward_guard"],
                dict(shape_env.var_to_range),
            )
            self._replace_once(gm, _relu_pattern, replacement)
            self.assertEqual(
                before,
                (
                    list(shape_env.pending_fresh_unbacked_symbols),
                    shape_env.counter["ignored_backward_guard"],
                    dict(shape_env.var_to_range),
                ),
            )
            self.assertTrue(fake_mode.allow_fallback_kernels)
            self.assertNotIn("val", self._node_with_target(gm, target).meta)

    def test_replace_pattern_meta_copy_disables_fallback_kernels(self):
        def replacement(x):
            return torch.ops.aten._foobar.default(x, False, False)

        gm = self._export_relu()
        fake_mode = next(
            node.meta["val"].fake_mode
            for node in gm.graph.nodes
            if node.op == "placeholder"
        )
        with mock.patch(
            "torch._subclasses.fake_tensor.run_fallback_kernel"
        ) as fallback:
            self._replace_once(gm, _relu_pattern, replacement)
        fallback.assert_not_called()
        self.assertTrue(fake_mode.allow_fallback_kernels)
        node = self._node_with_target(gm, torch.ops.aten._foobar.default)
        self.assertNotIn("val", node.meta)

    def test_replace_pattern_meta_execution_context_is_serialized(self):
        class BlockingLock:
            def __init__(self):
                self.attempted = Event()
                self.release = Event()

            def __enter__(self):
                self.attempted.set()
                self.release.wait()

            def __exit__(self, *args):
                pass

        shape_env = ShapeEnv()
        mode = FakeTensorMode(allow_fallback_kernels=True, shape_env=shape_env)
        lock = BlockingLock()

        def enter_context():
            with subgraph_rewriter._meta_execution_context(mode):
                pass

        with mock.patch.object(subgraph_rewriter, "_META_EXECUTION_LOCK", lock):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(enter_context)
                attempted = lock.attempted.wait(1)
                was_blocked = not future.done()
                lock.release.set()
                future.result()
        self.assertTrue(attempted)
        self.assertTrue(was_blocked)
        self.assertTrue(mode.allow_fallback_kernels)
        self.assertFalse(shape_env._error_on_new_guards)
        self.assertFalse(shape_env.frozen)

    def test_replace_pattern_meta_copy_skips_invalid_operator_inputs(self):
        def reshape_replacement(x):
            return torch.reshape(x, (3, 2))

        def prims_replacement(x):
            return torch.ops.prims.split_dim.default(x, 5, 2)

        for replacement, target in (
            (reshape_replacement, torch.reshape),
            (prims_replacement, torch.ops.prims.split_dim.default),
        ):
            gm = symbolic_trace(_ReluModel())
            placeholder = next(
                node for node in gm.graph.nodes if node.op == "placeholder"
            )
            placeholder.meta["val"] = torch.randn(4)
            subgraph_rewriter.replace_pattern(gm, _relu_pattern, replacement)
            replacement_node = self._node_with_target(gm, target)
            self.assertNotIn("val", replacement_node.meta)

    def test_replace_pattern_meta_copy_skips_mixed_fake_modes(self):
        def replacement(x, y):
            return torch.ops.aten.mul.Tensor(x, y)

        gm = symbolic_trace(_AddModel())
        placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
        self.assertEqual(len(placeholders), 2)
        for placeholder in placeholders:
            with FakeTensorMode() as mode:
                placeholder.meta["val"] = mode.from_tensor(torch.randn(4))

        self._replace_once(gm, _add_pattern, replacement)
        replacement_node = self._node_with_target(gm, torch.ops.aten.mul.Tensor)
        self.assertNotIn("val", replacement_node.meta)

    def test_replace_pattern_meta_copy_skips_unsupported_tensor_metadata(self):
        def replacement(x):
            return torch.ops.aten.neg.default(x)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "torch.quantize_per_tensor.*")
            warnings.filterwarnings("ignore", "The PyTorch API of nested tensors.*")
            quantized = torch.quantize_per_tensor(
                torch.randn(2, 3), scale=0.1, zero_point=0, dtype=torch.qint8
            )
            nested = torch.nested.nested_tensor([torch.randn(2), torch.randn(3)])

        for unsupported in (quantized, nested):
            gm = symbolic_trace(_ReluModel())
            placeholder = next(
                node for node in gm.graph.nodes if node.op == "placeholder"
            )
            placeholder.meta["val"] = unsupported
            self._replace_once(gm, _relu_pattern, replacement)
            replacement_node = self._node_with_target(gm, torch.ops.aten.neg.default)
            self.assertNotIn("val", replacement_node.meta)

    def test_replace_pattern_meta_copy_preserves_cross_input_aliases(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.neg.default(x), torch.ops.aten.neg.default(x)

        def replacement(x):
            return torch.ops.aten.view.default(x, [6])

        gm = symbolic_trace(M())
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        placeholder.meta["val"] = torch.randn(2, 3)
        matches = subgraph_rewriter.replace_pattern(gm, _neg_pattern, replacement)
        self.assertEqual(len(matches), 2)
        vals = [
            node.meta["val"]
            for node in gm.graph.nodes
            if node.target == torch.ops.aten.view.default
        ]
        self.assertEqual(
            vals[0].untyped_storage()._cdata, vals[1].untyped_storage()._cdata
        )

    def test_replace_pattern_meta_copy_does_not_mutate_existing_meta_val(self):
        def transpose_replacement(x):
            return torch.ops.aten.transpose_.default(x, 0, 1)

        def resize_replacement(x):
            return torch.ops.aten.resize_.default(x, [100])

        for replacement, target in [
            (transpose_replacement, torch.ops.aten.transpose_.default),
            (resize_replacement, torch.ops.aten.resize_.default),
        ]:
            with self.subTest(replacement=replacement.__name__):
                gm = self._export_relu()
                placeholder = next(
                    node for node in gm.graph.nodes if node.op == "placeholder"
                )
                original_val = placeholder.meta["val"]
                original_storage_size = original_val.untyped_storage().nbytes()

                self._replace_once(gm, _relu_pattern, replacement)
                self.assertEqual(original_val.shape, torch.Size([2, 3]))
                self.assertEqual(
                    original_val.untyped_storage().nbytes(), original_storage_size
                )
                replacement_node = self._node_with_target(gm, target)
                self.assertNotIn("val", replacement_node.meta)

    def test_replace_pattern_meta_copy_skips_existing_replacement_node(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.view.default(x, [6])

        def pattern(x):
            return torch.ops.aten.view.default(x, [6])

        def replacement(x):
            return x

        ep = torch.export.export(M(), (torch.randn(2, 3),))
        gm = ep.graph_module
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        original_shape = placeholder.meta["val"].shape

        self._replace_once(gm, pattern, replacement)
        self.assertEqual(placeholder.meta["val"].shape, original_shape)

    def test_matching_pattern_with_list_type_arg(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._reshape_alias_copy.default(x, [1, 2], [3, 4])

        def pattern(x, arg0, arg1):
            return torch.ops.aten._reshape_alias_copy.default(x, arg0, arg1)

        def replacement(x, arg0, arg1):
            return torch.ops.aten._reshape_alias_copy.default(x, arg1, arg0)

        traced = symbolic_trace(M())
        self._replace_once(traced, pattern, replacement)

        self.assertExpectedInline(
            traced.code.strip(),
            """\
def forward(self, x):
    _reshape_alias_copy_default_1 = torch.ops.aten._reshape_alias_copy.default(x, [3, 4], [1, 2]);  x = None
    return _reshape_alias_copy_default_1""",
        )

    def test_replacement_with_attrs(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1])
                self.b = torch.tensor([2])

            def forward(self, x):
                return x + self.a - self.b

        class Pattern(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1])

            def forward(self, x):
                return x + self.a

        class Replacement(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c = torch.tensor([3])

            def forward(self, x):
                return x - self.c

        traced = symbolic_trace(M())
        self._replace_once(traced, Pattern(), Replacement())

    def test_matching_variable_arguments(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.max_pool2d_with_indices.default(
                    x, [2, 2], stride=[2, 2]
                )

        def pattern(x, kernel_size, stride):
            # default padding is [0, 0]
            return torch.ops.aten.max_pool2d_with_indices.default(
                x, kernel_size, stride, padding=[0, 0]
            )

        traced = symbolic_trace(M())
        self._replace_once(traced, pattern, pattern)

    def test_replaced_nodes(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.add(x, y)

        def pattern(x, y):
            return torch.add(x, y)

        def replacement(x, y):
            return torch.sub(torch.mul(x, y), y)

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced, pattern, replacement
        )

        def check_replacement_nodes(self, traced, matches):
            replacement_nodes_in_graph = [
                node
                for node in traced.graph.nodes
                if node.target in {torch.sub, torch.mul}
            ]
            replacement_nodes_in_res = [r for m in matches for r in m.replacements]
            self.assertEqual(
                len(replacement_nodes_in_graph), len(replacement_nodes_in_res)
            )
            self.assertEqual(replacement_nodes_in_graph, replacement_nodes_in_res)
            return len(replacement_nodes_in_graph)

        self.assertEqual(check_replacement_nodes(self, traced, matches), 2)

    def test_replace_pattern_with_callback(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.add(x, y)

        def pattern(x, y):
            return torch.add(x, y)

        def replacement(x, y):
            return torch.sub(torch.mul(x, y), y)

        traced = symbolic_trace(M())
        # Return the same replacement graph for all matches, but have it be a unique
        # object each time.
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced,
            pattern,
            replacement_callback=lambda *args: symbolic_trace(replacement).graph,
        )

        def check_replacement_nodes(self, traced, matches):
            replacement_nodes_in_graph = [
                node
                for node in traced.graph.nodes
                if node.target in {torch.sub, torch.mul}
            ]
            replacement_nodes_in_res = [r for m in matches for r in m.replacements]
            self.assertEqual(
                len(replacement_nodes_in_graph), len(replacement_nodes_in_res)
            )
            self.assertEqual(replacement_nodes_in_graph, replacement_nodes_in_res)
            return len(replacement_nodes_in_graph)

        self.assertEqual(check_replacement_nodes(self, traced, matches), 2)
