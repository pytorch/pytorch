# Owner(s): ["module: inductor"]

from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.compile_fx import _compile_fx_inner, get_cpp_wrapper_config
from torch._inductor.test_case import TestCase as InductorTestCase


class DummyOutput:
    def post_compile(self, example_inputs, constants, graph_kwargs) -> None:
        self.graph_kwargs = graph_kwargs


class AotModeTest(InductorTestCase):
    def test_compile_fx_inner_uses_explicit_aot_mode(self):
        def fn(x):
            return (x,)

        gm = torch.fx.symbolic_trace(fn)
        gm.shape_env = None
        example_inputs = (torch.randn(2),)
        dummy_output = DummyOutput()
        seen_kwargs = None

        def fake_codegen_and_compile(*args, **kwargs):
            nonlocal seen_kwargs
            seen_kwargs = kwargs
            return dummy_output

        with (
            config.patch(
                {
                    "force_disable_caches": False,
                    "fx_graph_cache": False,
                    "fx_graph_remote_cache": False,
                }
            ),
            mock.patch(
                "torch._inductor.compile_fx.fx_codegen_and_compile",
                side_effect=fake_codegen_and_compile,
            ),
        ):
            result = _compile_fx_inner(gm, example_inputs, aot_mode=True)

        self.assertIs(result, dummy_output)
        self.assertIsNotNone(seen_kwargs)
        self.assertTrue(seen_kwargs["aot_mode"])
        self.assertTrue(dummy_output.graph_kwargs["aot_mode"])

    def test_cpp_wrapper_config_uses_explicit_aot_mode(self):
        with (
            config.patch(
                {
                    "graph_partition": False,
                    "triton.autotune_at_compile_time": None,
                    "triton.cudagraphs": True,
                }
            ),
            mock.patch("torch._inductor.compile_fx.has_triton", return_value=True),
        ):
            aot_config = get_cpp_wrapper_config(aot_mode=True)
            jit_config = get_cpp_wrapper_config(aot_mode=False)

        self.assertTrue(aot_config["triton.autotune_at_compile_time"])
        self.assertFalse(aot_config["triton.cudagraphs"])
        self.assertFalse(jit_config["triton.autotune_at_compile_time"])
        self.assertTrue(jit_config["triton.cudagraphs"])


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
