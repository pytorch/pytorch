# Owner(s): ["module: dynamo"]

import os

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._functorch._aot_autograd
from torch._dynamo.utils import counters
from torch._functorch import config
from torch._functorch._aot_autograd.autograd_cache import (
    AOTAutogradCache,
    autograd_cache_hash,
    BypassAOTAutogradCache,
    check_cacheable,
)
from torch._functorch._aot_autograd.schemas import AOTConfig


torch._inductor.config.fx_graph_cache = True


class AOTAutogradCacheTests(torch._dynamo.test_case.TestCase):
    def setUp(self):
        """
        Reset all counters and caches before each unit test
        """
        super().setUp()
        counters.clear()
        self._clear_all_caches()

    def _clear_all_caches(self):
        """
        Clear every cache, including AOTAutgradCache and FXCache
        """
        torch._inductor.codecache.FxGraphCache.clear()
        AOTAutogradCache.clear()
        self._clear_extra_caches()

    def _clear_extra_caches(self):
        """
        Clear unrelated caches, like dynamo and PyCodeCache
        """
        torch._dynamo.reset()
        for m in torch._inductor.codecache.PyCodeCache.cache.values():
            os.remove(m.__file__)
        torch._inductor.codecache.PyCodeCache.cache_clear()

    @config.patch({"enable_aot_autograd_cache": True})
    def test_basic(self):
        """
        Verify the interactions between FXGraphCache and AOTAutogradCache.
        """

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        self._clear_extra_caches()
        self.assertEqual(fn(a, b), compiled_fn(a, b))

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

    @config.patch({"enable_aot_autograd_cache": True})
    def test_autograd_function(self):
        """
        Autograd functions are not supported in the cache yet
        """

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25, requires_grad=True)
        b = torch.rand(5, 5, requires_grad=True)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

        # Second call should hit FXGraphCache, miss AOTAutogradCache
        self._clear_extra_caches()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)


class AOTAutogradCachePicklerTests(torch._dynamo.test_case.TestCase):
    """
    Tests for cache key generation
    """

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

        def compiler(gm, inputs, **kwargs):
            nonlocal fx_graph
            fx_graph = gm
            return gm

        g = torch.compile(fn, backend=compiler, fullgraph=True)
        result = g(*args, **kwargs)
        return (result, fx_graph)

    def gen_cache_key(self, f, config, inputs=None):
        if inputs is None:
            inputs = [torch.randn(3)]
        _, fx_g = self._get_dynamo_output(f, *inputs)
        check_cacheable(fx_g)
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
