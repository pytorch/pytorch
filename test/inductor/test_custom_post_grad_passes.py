# Owner(s): ["module: inductor"]
import contextlib

import torch
import torch._inductor.pattern_matcher as pattern_matcher

from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters

from torch._inductor import config
from torch._inductor.lowering import lowerings as L
from torch._inductor.pattern_matcher import Arg, CallFunction, PatternMatcherPass

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
mkldnn = torch.ops.mkldnn


class TestPostGradCustomPrePostPass(TestCustomPassBase):
    #  mkldnn fusion's pattern_matcher
    # (torch/_inductor/fx_passes/mkldnn_fusion.py),
    # and apply it to custom post_grad_passes.
    def _register_mkldnn_conv_relu_fusion(self, custom_pass_dict):
        # pattern
        def _mkldnn_conv_relu_pattern():
            return CallFunction(
                aten.relu,
                CallFunction(
                    mkldnn._convolution_pointwise.default,
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
                return L[mkldnn._convolution_pointwise.default](*computation_args)

            return fn

        _register_fusion_lowering(_mkldnn_conv_relu_pattern(), custom_pass_dict)

    # custom post grad pass
    class _CustomPass(PatternMatcherPass):
        def __init__(self):
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

    def test_custom_pre_pass(self):
        # leave custom pass only in post_grad_passes()
        dafault_pattern_matcher = config.pattern_matcher
        config.pattern_matcher = False
        # define pattern match as custom post grad opt pass
        config.post_grad_custom_pre_pass = self._CustomPass()
        config.post_grad_custom_post_pass = None
        # init mkldnn fusion on custom_matcher
        self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_pre_pass)

        mod = self._ConvReLU(16, 16).eval()
        x = torch.randn((1, 16, 56, 56), dtype=torch.float32)

        match_count = 1
        match_nodes = 2
        other_match_count = 1  # conv prepack weight
        other_match_nodes = 1  # conv prepack weight
        self._test_common(
            mod, (x,), match_count + other_match_count, match_nodes + other_match_nodes
        )

        # restore default pattern_matcher
        config.pattern_matcher = dafault_pattern_matcher

    def test_custom_post_pass(self):
        # leave custom pass only in post_grad_passes()
        dafault_pattern_matcher = config.pattern_matcher
        config.pattern_matcher = False
        # define pattern match as custom post grad opt pass
        config.post_grad_custom_pre_pass = None
        config.post_grad_custom_post_pass = self._CustomPass()
        # init mkldnn fusion on custom_matcher
        self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_post_pass)

        mod = self._ConvReLU(16, 16).eval()
        x = torch.randn((1, 16, 56, 56), dtype=torch.float32)

        match_count = 1
        match_nodes = 2
        other_match_count = 1  # conv prepack weight
        other_match_nodes = 1  # conv prepack weight
        self._test_common(
            mod, (x,), match_count + other_match_count, match_nodes + other_match_nodes
        )

        # restore default pattern_matcher
        config.pattern_matcher = dafault_pattern_matcher


if __name__ == "__main__":
    if IS_LINUX and HAS_CPU and torch.backends.mkldnn.is_available():
        run_tests()
