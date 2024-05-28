# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo
import torch._dynamo.test_case

import torch._functorch._aot_autograd
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.autograd_cache import (
    autograd_cache_hash,
    BypassAOTAutogradCache,
)
from torch._functorch._aot_autograd.schemas import AOTConfig
from torch._inductor import config as inductor_config


class AOTAutogradCachePicklerTests(torch._dynamo.test_case.TestCase):
    @property
    def device_type(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def default_config(self):
        return AOTConfig(
            fw_compiler=None,
            bw_compiler=None,
            inference_compiler=None,
            partition_fn=None,
            decompositions={},
            num_params_buffers=0,
            aot_id=0,
            keep_inference_input_mutations=False,
            dynamic_shapes=True,
            aot_autograd_arg_pos_to_source=None,
            is_export=False,
            no_tangents=False,
            enable_log=False,
        )

    def _get_dynamo_output(self, fn, *args, **kwargs):
        # Reset dynamo between runs
        torch._dynamo.reset()
        fx_graph = None
        example_inputs = None

        def compiler(gm, inputs, **kwargs):
            nonlocal fx_graph
            nonlocal example_inputs
            fx_graph = gm
            example_inputs = inputs
            return gm

        g = torch.compile(fn, backend=compiler, fullgraph=True)
        result = g(*args, **kwargs)
        return (result, fx_graph, example_inputs)

    def gen_cache_key(self, f, config, inputs=None):
        if inputs is None:
            inputs = [torch.ones(3)]
        _, fx_g, example_inputs = self._get_dynamo_output(f, *inputs)
        return autograd_cache_hash(fx_g, example_inputs, config)

    def test_basic_hash_key(self):
        def fn(x):
            return x.sin().cos()

        config = self.default_config()
        # Check hash is stable on multiple runs
        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config)
        self.assertEqual(c1, c2)

    def test_identical_graphs_and_configs(self):
        def fn(x):
            return x.sin().cos()

        def fn2(x):
            y = x.sin()
            z = y.cos()
            return z

        # Make the id different, but otherwise identical
        config = self.default_config()
        config2 = self.default_config()
        config2.aot_id = 1

        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config2)
        self.assertEqual(c1, c2)

    def test_different_graphs(self):
        def fn(x):
            return x.cos().sin()

        def fn2(x):
            return x.sin().cos()

        config = self.default_config()
        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn2, config)
        self.assertNotEqual(c1, c2)

    def test_different_configs(self):
        def fn(x):
            return x.cos().sin()

        config = self.default_config()
        config2 = self.default_config()
        config2.dynamic_shapes = False
        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config2)
        self.assertNotEqual(c1, c2)

    def test_different_inputs(self):
        def fn(x):
            return x.cos().sin()

        config = self.default_config()
        c1 = self.gen_cache_key(fn, config, inputs=[torch.ones(3)])
        c2 = self.gen_cache_key(fn, config, inputs=[torch.ones(2)])
        self.assertNotEqual(c1, c2)

    def test_different_global_configs(self):
        def fn(x):
            return x.cos().sin()

        config = self.default_config()

        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config)
        self.assertEqual(c1, c2)

        c1 = self.gen_cache_key(fn, config)

        # Change functorch config
        with functorch_config.patch(
            {"debug_assert": not functorch_config.debug_assert}
        ):
            c2 = self.gen_cache_key(fn, config)

        self.assertNotEqual(c1, c2)

        c1 = self.gen_cache_key(fn, config)
        # Change inductor config
        with inductor_config.patch({"debug": not inductor_config.debug}):
            c2 = self.gen_cache_key(fn, config)

        self.assertNotEqual(c1, c2)

        c1 = self.gen_cache_key(fn, config)
        # Change torch grad enabled
        with torch.no_grad():
            c2 = self.gen_cache_key(fn, config)
        self.assertNotEqual(c1, c2)

    def test_incompatible_function(self):
        @torch._dynamo.allow_in_graph
        class AllowInGraphFunc(torch.autograd.Function):
            @staticmethod
            def forward(_, x):
                torch._dynamo.graph_break()
                return x.sin()

        def fn(x):
            return AllowInGraphFunc.apply(x)

        config = self.default_config()
        self.assertRaises(
            BypassAOTAutogradCache, lambda: self.gen_cache_key(fn, config)
        )

    def test_normal_torch_function(self):
        @torch._dynamo.allow_in_graph
        def fn(x):
            y = torch.sin(x)
            z = torch.cos(x)
            w = y + z
            w.abs()
            return w

        config = self.default_config()
        self.gen_cache_key(fn, config)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
