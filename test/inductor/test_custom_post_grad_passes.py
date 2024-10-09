# Owner(s): ["module: inductor"]
import contextlib
import operator
from collections import defaultdict

import torch
import torch._inductor.pattern_matcher as pattern_matcher
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.lowering import lowerings as L
from torch._inductor.pattern_matcher import Arg, CallFunction, PatternMatcherPass
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CPU


@config.patch({"freezing": True})
class TestCustomPassBase(TestCase):
    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return tuple(clone(x) for x in inputs)

    def _test_common(
        self,
        mod,
        inputs,
        matcher_count,
        matcher_nodes,
        atol=1e-5,
        rtol=1.3e-6,
    ):
        counters.clear()
        maybe_autocast = contextlib.nullcontext()
        with torch.no_grad(), maybe_autocast:
            clone_inputs = self._clone_inputs(inputs)
            expected = mod(*inputs)
            actual = torch.compile(mod)(*clone_inputs)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            self.assertEqual(
                counters["inductor"]["pattern_matcher_count"], matcher_count
            )
            self.assertEqual(
                counters["inductor"]["pattern_matcher_nodes"],
                matcher_nodes,
            )


aten = torch.ops.aten
onednn = torch.ops.onednn


def change_cos_pass(graph):
    for node in graph.nodes:
        if node.op == "call_function" and node.target == aten.cos.default:
            node.target = aten.sin.default


class TestPostGradCustomPrePostPass(TestCustomPassBase):
    #  onednn fusion's pattern_matcher
    # (torch/_inductor/fx_passes/onednn_fusion.py),
    # and apply it to custom post_grad_passes.
    def _register_mkldnn_conv_relu_fusion(self, custom_pass_dict):
        # pattern
        def _mkldnn_conv_relu_pattern():
            return CallFunction(
                aten.relu,
                CallFunction(
                    onednn._convolution_pointwise.default,
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    _users=1,
                ),
            )

        # utils of pattern matcher registration
        def _register_fusion_lowering(pattern, custom_pass_dict):
            def dummy_check(m):
                return True

            def register_custom_lowering_pattern(
                pattern, extra_check, custom_pass_dict
            ):
                return pattern_matcher.register_lowering_pattern(
                    pattern, extra_check, pass_dict=custom_pass_dict
                )

            @register_custom_lowering_pattern(pattern, dummy_check, custom_pass_dict)
            def fn(match, *args, **kwargs):
                computation_args = list(args)[:-3] + ["relu", [], ""]
                return L[onednn._convolution_pointwise.default](*computation_args)

            return fn

        _register_fusion_lowering(_mkldnn_conv_relu_pattern(), custom_pass_dict)

    # custom post grad pass
    class _CustomPass(PatternMatcherPass):
        def __init__(self) -> None:
            super().__init__()

        def __call__(self, g: torch.fx.graph.Graph):
            self.apply(g)

    # case model
    class _ConvReLU(torch.nn.Module):
        def __init__(self, ic, oc):
            super().__init__()
            self.conv = torch.nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x1 = self.conv(x)
            return x1.relu()

    def test_custom_joint_pass_pre(self):
        with config.patch(joint_custom_pre_pass=change_cos_pass):

            def g(x):
                return x.sin().sin().sin()

            def f(x):
                return x.cos().cos().cos()

            x = torch.randn(8, dtype=torch.float32)
            torch.testing.assert_close(torch.compile(f)(x), g(x))

    def test_custom_joint_pass_post(self):
        with config.patch(joint_custom_post_pass=change_cos_pass):

            def g(x):
                return x.sin().sin().sin()

            def f(x):
                return x.cos().cos().cos()

            x = torch.randn(8, dtype=torch.float32)
            torch.testing.assert_close(torch.compile(f)(x), g(x))

    def test_custom_pre_pass(self):
        with config.patch(
            # leave custom pass only in post_grad_passes()
            pattern_matcher=False,
            post_grad_custom_pre_pass=self._CustomPass(),
            # define pattern match as custom post grad opt pass
            post_grad_custom_post_pass=None,
        ):
            # init onednn fusion on custom_matcher
            self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_pre_pass)

            mod = self._ConvReLU(16, 16).eval()
            x = torch.randn((1, 16, 56, 56), dtype=torch.float32)

            match_count = 1
            match_nodes = 2
            other_match_count = 1  # conv prepack weight
            other_match_nodes = 1  # conv prepack weight
            self._test_common(
                mod,
                (x,),
                match_count + other_match_count,
                match_nodes + other_match_nodes,
            )

    def test_custom_post_pass(self):
        with config.patch(
            # leave custom pass only in post_grad_passes()
            pattern_matcher=False,
            # define pattern match as custom post grad opt pass
            post_grad_custom_pre_pass=None,
            post_grad_custom_post_pass=self._CustomPass(),
        ):
            # init onednn fusion on custom_matcher
            self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_post_pass)

            mod = self._ConvReLU(16, 16).eval()
            x = torch.randn((1, 16, 56, 56), dtype=torch.float32)

            match_count = 1
            match_nodes = 2
            other_match_count = 1  # conv prepack weight
            other_match_nodes = 1  # conv prepack weight
            self._test_common(
                mod,
                (x,),
                match_count + other_match_count,
                match_nodes + other_match_nodes,
            )

    def test_custom_pre_grad_pass(self):
        saved_graph = [None]

        def merge_mm_shared_rhs(graph: fx.Graph):
            """
            Bad POC of merging mm with a shared RHS.
            i.e. [mm(x, W), mm(x2, W)] => mm(cat(x, x2), W).split()

            Isn't actually safe for a couple reasons. For example, it doesn't handle the
            case where the LHS inputs depend on each other
            """
            saved_graph[0] = graph
            matmuls = [n for n in graph.nodes if n.target == torch.mm]
            rhs_vals = defaultdict(set)
            for m in matmuls:
                rhs_vals[m.args[1]].add(m)

            order = {}
            for idx, n in enumerate(graph.nodes):
                order[n] = idx

            for rhs, matmuls in rhs_vals.items():
                if len(matmuls) == 1:
                    continue
                matmuls = sorted(matmuls, key=lambda x: order[x])
                with graph.inserting_before(matmuls[0]):
                    lhs_vals = [m.args[0] for m in matmuls]
                    new_cat = graph.create_node(
                        "call_function", torch.cat, args=(lhs_vals, 0)
                    )
                    new_mm = graph.create_node(
                        "call_function", torch.mm, args=(new_cat, rhs)
                    )
                    split_vals = graph.create_node(
                        "call_function",
                        torch.split,
                        args=(
                            new_mm,
                            [l.meta["example_value"].shape[0] for l in lhs_vals],
                        ),
                    )
                for idx, m in enumerate(matmuls):
                    m.target = operator.getitem
                    m.args = (split_vals, idx)

        @config.patch(pre_grad_custom_pass=merge_mm_shared_rhs)
        def inner_test():
            @torch.compile
            def f(W, nested_seqs):
                outs = [torch.mm(s, W) for s in nested_seqs]
                return outs

            W = torch.randn(16, 16, dtype=torch.bfloat16)
            nested_seqs = [
                torch.randn(l, 16, dtype=torch.bfloat16) for l in [4, 8, 5, 3]
            ]

            f(W, nested_seqs)
            assert saved_graph[0] is not None
            matmuls = [n for n in saved_graph[0].nodes if n.target == torch.mm]
            assert len(matmuls) == 1

        inner_test()


if __name__ == "__main__":
    if IS_LINUX and HAS_CPU and torch.backends.onednn.is_available():
        run_tests()
