# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo
import torch._dynamo.test_case

import torch._functorch._aot_autograd
from torch._functorch._aot_autograd.autograd_cache import autograd_cache_hash
from torch._functorch._aot_autograd.schemas import AOTConfig


class AOTAutogradCachePicklerTests(torch._dynamo.test_case.TestCase):
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
        fx_graph = None

        def compiler(gm, inputs, **kwargs):
            nonlocal fx_graph
            fx_graph = gm
            return gm

        g = torch.compile(fn, backend=compiler)
        result = g(*args, **kwargs)
        return (result, fx_graph)

    def gen_cache_key(self, f, config, inputs=None):
        if inputs is None:
            inputs = [torch.randn(3)]
        _, fx_g = self._get_dynamo_output(f, *inputs)
        return autograd_cache_hash(fx_g, config)

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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
