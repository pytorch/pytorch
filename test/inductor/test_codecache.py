# Owner(s): ["module: inductor"]
import os
import pickle
import shutil
import tempfile
import unittest
from typing import List, Optional, Union
from unittest import mock

import torch
from torch._dynamo import reset
from torch._dynamo.utils import counters
from torch._inductor import config, metrics
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codecache import (
    cuda_compile_command,
    CUDACodeCache,
    FxGraphCachePickler,
    FxGraphHashDetails,
    PyCodeCache,
    TensorMetadata,
    TensorMetadataAndValues,
)
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch._inductor.graph import GraphLowering
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import clear_inductor_caches, fresh_inductor_cache
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA,
    HAS_GPU,
    HAS_MULTIGPU,
    HAS_TRITON,
    requires_gpu,
    requires_triton,
)
from torch.testing._internal.triton_utils import requires_cuda


try:
    from .mock_cache import global_stats, PatchCaches, Stats
except ImportError:
    from mock_cache import global_stats, PatchCaches, Stats  # @manual


if HAS_TRITON:
    import triton  # @manual

    from torch.testing._internal.triton_utils import add_kernel

torch._dynamo.config.fake_tensor_cache_enabled = True
torch._dynamo.config.fake_tensor_cache_crosscheck_enabled = True


class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)

    def forward(self, inp):
        return self.fc1(inp)


def _run_codecache_test(start_method):
    with torch._inductor.config.patch(
        worker_start_method=start_method, compile_threads=16
    ):
        AsyncCompile.warm_pool()

        model = MyModel().to(device=GPU_TYPE)
        model = torch.compile(model)
        inp = torch.rand(10, 10).to(device=GPU_TYPE)
        model(inp).sum().backward()


@requires_gpu()
def test_codecache_spawn():
    _run_codecache_test("spawn")


@requires_gpu()
def test_codecache_fork():
    _run_codecache_test("fork")


class MyModelConv2d(torch.nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, dim, kernel_size=3, stride=2, bias=False)
        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        torch._dynamo.graph_break()
        x = self.conv2(x)
        return x


@instantiate_parametrized_tests
class TestFxGraphCache(TestCase):
    device_type = GPU_TYPE

    def setUp(self):
        super().setUp()
        counters.clear()
        PatchCaches.setUp()

    def tearDown(self):
        super().tearDown()
        PatchCaches.tearDown()

    def reset(self):
        torch._dynamo.reset()
        clear_inductor_caches()

    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    @parametrize("bundle_triton", (False, True))
    def test_cache_load_function(self, device, dtype, dynamic, bundle_triton):
        """
        Verify that we can populate and load functions from the cache.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25, dtype=dtype, device=device)
        b = torch.rand(5, 5, dtype=dtype, device=device)

        with config.patch(bundle_triton_into_fx_graph_cache=bundle_triton):
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            # A first call should miss in the cache.
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

            if bundle_triton and device != "cpu":
                self.assertEqual(counters["inductor"]["triton_bundler_save_kernel"], 7)
                self.assertEqual(counters["inductor"]["triton_bundler_write_kernel"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            for m in torch._inductor.codecache.PyCodeCache.cache.values():
                os.remove(m.__file__)
            # Clean triton kernels
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)
            self.reset()
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

            if bundle_triton and device != "cpu":
                self.assertEqual(counters["inductor"]["triton_bundler_save_kernel"], 7)
                self.assertEqual(counters["inductor"]["triton_bundler_write_kernel"], 7)

    @requires_triton()
    @config.patch({"fx_graph_remote_cache": True})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    @parametrize("bundle_triton", (False, True))
    def test_remote_cache_load_function(self, device, dtype, dynamic, bundle_triton):
        from unittest.mock import patch

        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25, dtype=dtype, device=device)
        b = torch.rand(5, 5, dtype=dtype, device=device)

        with config.patch(
            {
                "fx_graph_remote_cache": True,
                "bundle_triton_into_fx_graph_cache": bundle_triton,
            }
        ), patch.dict(os.environ), PatchCaches():
            for _ in range(4):
                with fresh_inductor_cache():
                    compiled_fn = torch.compile(fn, dynamic=dynamic)
                    self.assertEqual(fn(a, b), compiled_fn(a, b))
                reset()

        self.assertEqual(global_stats.fx_graph, Stats(1, 3, 1))

        if config.is_fbcode():
            # Check that the cache entries seem reasonable
            for k in global_stats.fx_graph.cache.keys():
                self.assertRegex(k, r"pt2:fx-graph-v1::[0-9a-z]{52}:c10")

    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.float64))
    @parametrize("dynamic", (False, True))
    def test_cache_load_model(self, device, dtype, dynamic):
        """
        Verify that we can populate and load models from the cache.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        def fn(mod, x):
            mod.zero_grad()
            mod(x).sum().backward()
            return [p.grad for p in mod.parameters()]

        compiled_fn = torch.compile(fn, dynamic=dynamic)

        mod = MyModelConv2d().to(device=device, dtype=dtype)
        inp = torch.randn(2, 3, 16, 16, device=device, dtype=dtype)

        # The first call should see all cache misses.
        counters.clear()
        grads1 = compiled_fn(mod, inp)
        self.assertGreater(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # The second should see all hits. (First reset so in-memory guards
        # don't prevent compilation).
        counters.clear()
        self.reset()
        grads2 = compiled_fn(mod, inp)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], 0)

        # And the results should be the same.
        self.assertEqual(grads1, grads2)

    @largeTensorTest("64GB", device=GPU_TYPE)
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE,))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_cache_load_with_guards_int32_bounds(self, device, dtype):
        """
        Test caching the same graph, but under conditions that introduce guards
        for tensor sizes < int32.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires CUDA SM80 or later")

        def fn(x, y):
            return (x + x, y + y)

        compiled_fn = torch.compile(fn, dynamic=True)

        # Iterate over different shapes, varying whether the total
        # size is below or above int32. For each combination, we expect
        # different guards around whether the symbolic sizes do or do
        # not exceed int32.
        shapes = (
            ((5, 6), (7, 8)),
            ((5, 6), (47000, 47001)),
            ((47000, 47001), (5, 6)),
        )
        for a_shape, b_shape in shapes:
            a = torch.rand(a_shape, device=device, dtype=dtype)
            b = torch.rand(b_shape, device=device, dtype=dtype)

            # AVOID a dynamo reset here. We expect guards to have been
            # added that will be violated with the new shape. We should
            # see a recompilation (along with a cache miss).
            counters.clear()
            res1 = compiled_fn(a, b)
            self.assertGreater(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            # A second call should hit. (Reset here to force compilation).
            counters.clear()
            self.reset()
            res2 = compiled_fn(a, b)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], 0)

            self.assertEqual(res1, res2)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    def test_cache_load_with_guards_static_bounds(self, device, dtype):
        """
        Test caching the same graph, but under conditions that introduce guards
        for static bounds.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        # See lowering; for all of the pooling operators, we always guard and
        # make the height/width static.
        def fn(x):
            return torch.nn.functional.adaptive_avg_pool2d(x, [5, 7])

        compiled_fn = torch.compile(fn, dynamic=True)

        # Iterate over different input shapes. Each new shape should cause
        # a cache miss.
        shapes = ((1, 64, 8, 9), (1, 64, 9, 10), (1, 64, 10, 11))
        for shape in shapes:
            x = torch.rand(shape, device=device, dtype=dtype)

            # AVOID a dynamo reset here. For each cache hit, we expect guards
            # to have been added that will be violated with each new shape.
            # We should see a recompilation (along with a cache miss).
            counters.clear()
            res1 = compiled_fn(x)
            self.assertGreater(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            # A second call should hit.
            counters.clear()
            self.reset()
            res2 = compiled_fn(x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], 0)

            self.assertEqual(res1, res2)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE, "cpu"))
    def test_constant_handling(self, device):
        """
        Test that different constants are recognized correctly.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        def fn1(x):
            return x + torch.tensor(list(range(0, 12)), device=device)

        def fn2(x):
            return x + torch.tensor(list(range(1, 13)), device=device)

        a = torch.rand(12, device=device)

        compiled_fn1 = torch.compile(fn1)
        compiled_fn2 = torch.compile(fn2)

        # A call to fn1 should miss in the cache.
        self.assertEqual(fn1(a), compiled_fn1(a))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # A call to fn2 should also miss (the constant is different)
        self.assertEqual(fn2(a), compiled_fn2(a))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

    @requires_cuda
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_flex_attention_caching(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        block_mask = create_block_mask(
            lambda b, h, q, kv: q >= kv, None, None, 2048, 2048
        )

        def score_mod(score, b, h, q, kv):
            return score + (q - kv)

        def fn(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

        def score_mod2(score, b, h, q, kv):
            return score

        def fn2(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod2, block_mask=block_mask)

        a, b, c = (torch.randn(1, 4, 512, 64).cuda() for _ in range(3))
        compiled_fn = torch.compile(fn)
        compiled_fn2 = torch.compile(fn2)

        atol, rtol = 1e-4, 1e-4

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b, c), compiled_fn(a, b, c), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        for m in torch._inductor.codecache.PyCodeCache.cache.values():
            os.remove(m.__file__)
        self.reset()
        self.assertEqual(fn(a, b, c), compiled_fn(a, b, c), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

        # A third call with different score_mod should have a cache miss
        for m in torch._inductor.codecache.PyCodeCache.cache.values():
            os.remove(m.__file__)
        self.reset()
        self.assertEqual(fn2(a, b, c), compiled_fn2(a, b, c), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @requires_gpu()
    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_triton_higher_order_op_bypass(self):
        """
        Verify that we bypass the cache when we have a triton higher order ops.
        """

        def fn(x, y):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (  # noqa: E731
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
            )
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4)
            return output

        compiled_fn = torch.compile(fn, fullgraph=True)

        x = torch.randn(4, device=GPU_TYPE)
        y = torch.randn(4, device=GPU_TYPE)
        compiled_fn(x, y)

        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertGreater(counters["inductor"]["fxgraph_cache_bypass"], 0)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_generated_kernel_count(self):
        """
        Test that we bump the generated_kernel_count metric on a cache hit.
        """

        def fn(x, y):
            return (x * y + y,)

        a = torch.rand(5, 5)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn)

        metrics.reset()
        self.assertEqual(metrics.generated_kernel_count, 0)

        # Verify the "miss" case.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertEqual(metrics.generated_kernel_count, 1)

        # Verify the "hit" case
        self.reset()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(metrics.generated_kernel_count, 2)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_inductor_counters(self):
        """
        Test that we bump the inductor counters on a cache hit.
        """
        compile_to_fn = GraphLowering.compile_to_fn

        counter_name = "a_test_counter"
        counter_incr = 7

        def bump_counter(self):
            # Mock that bumps some arbitrary test counter by a set amount, then calls
            # the original GraphLowering.compile_to_fn.
            counters["inductor"][counter_name] += counter_incr
            return compile_to_fn(self)

        with mock.patch.object(GraphLowering, "compile_to_fn", bump_counter):

            def fn(a, b):
                return torch.mm(a, b)

            a = torch.rand(8, 32, device="cpu")
            b = torch.rand(32, 8, device="cpu")

            compiled_fn = torch.compile(fn)

            # Verify the "miss" case.
            counter_val = 2
            counters["inductor"][counter_name] = counter_val
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(
                counters["inductor"][counter_name], counter_val + counter_incr
            )

            # Verify the "hit" case.
            self.reset()
            counter_val = 5
            counters["inductor"][counter_name] = counter_val
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(
                counters["inductor"][counter_name], counter_val + counter_incr
            )

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_cache_clear(self):
        """
        Test clearing the cache.
        """

        def fn(x, y):
            return (x * y,)

        a = torch.rand(5, 5)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn)

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # A second call should hit.
        counters.clear()
        self.reset()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        # Clear the cache; now we should miss.
        counters.clear()
        self.reset()
        torch._inductor.codecache.FxGraphCache.clear()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_cache_with_nt(self):
        def gen_nt(r):
            values = torch.randn(r, 16)
            offsets = torch.tensor([0, 2, 3, 6, 13, r])
            return torch.nested.nested_tensor_from_jagged(values, offsets)

        def fn(nt):
            if nt.values().size(0) % 16 == 0:
                return nt.sin()
            return nt.cos()

        inp1 = gen_nt(19)
        inp2 = gen_nt(20)

        counters.clear()
        torch.compile(fn)(inp1)
        torch.compile(fn)(inp2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        self.reset()
        counters.clear()
        torch.compile(fn)(inp1)
        torch.compile(fn)(inp2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_cache_with_symint_non_arg_guard(self):
        def fn(x, ref_id):
            self_id = 22
            if self_id == ref_id:
                x = torch.mul(x, 1.0)
            else:
                x = torch.mul(x, 0)
            return x

        x = torch.ones(2)

        counters.clear()
        torch.compile(fn, fullgraph=True, dynamic=True)(x, 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        self.reset()
        counters.clear()
        torch.compile(fn, fullgraph=True, dynamic=True)(x, 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_cache_guard(self):
        def f(x, val):
            if val > 5:
                return x.sin()
            else:
                return x.cos()

        x = torch.ones(2)
        a = torch.compile(f, dynamic=True)(x, 6)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        self.reset()
        counters.clear()
        b = torch.compile(f, dynamic=True)(x, 4)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        self.assertNotEqual(a, b)


class TestFxGraphCacheHashing(TestCase):
    def test_tensor_constants(self):
        """
        Test the hashing of tensor constants.
        """
        data = FxGraphCachePickler.dumps(torch.tensor(list(range(9))))
        self.assertIsInstance(pickle.loads(data), TensorMetadataAndValues)

    def test_hash_fake_tensors(self):
        """
        Test hashing (pickling) FakeTensors with various characteristics.
        """
        with torch._subclasses.FakeTensorMode():
            # Verify that FakeTensors get pickled into a TensorMetadata:
            data = FxGraphCachePickler.dumps(torch.randn(1))
            self.assertIsInstance(pickle.loads(data), TensorMetadata)

            # Different shapes:
            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3)),
                FxGraphCachePickler.dumps(torch.randn(3)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3)),
                FxGraphCachePickler.dumps(torch.randn(4)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3)),
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
            )

            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
                FxGraphCachePickler.dumps(torch.randn(3, 4)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
                FxGraphCachePickler.dumps(torch.randn(4, 3)),
            )

            # Different strides:
            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
                FxGraphCachePickler.dumps(
                    torch.randn(3, 3).transpose(0, 1).transpose(0, 1)
                ),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
                FxGraphCachePickler.dumps(torch.randn(3, 3).transpose(0, 1)),
            )

            # Different storage offsets:
            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3)[1:]),
                FxGraphCachePickler.dumps(torch.randn(3)[1:]),
            )
            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3)[1:]),
                FxGraphCachePickler.dumps(torch.randn(2)),
            )

            # Different dtypes:
            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float32)),
                FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float32)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float32)),
                FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float64)),
            )

            # Different 'requires_grad':
            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3, requires_grad=True)),
                FxGraphCachePickler.dumps(torch.randn(3, requires_grad=True)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, requires_grad=True)),
                FxGraphCachePickler.dumps(torch.randn(3, requires_grad=False)),
            )

            # Different memory formats:
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(1, 2, 3, 4)),
                FxGraphCachePickler.dumps(
                    torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last)
                ),
            )

            # Different devices:
            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3, device="meta")),
                FxGraphCachePickler.dumps(torch.randn(3, device="meta")),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, device="meta")),
                FxGraphCachePickler.dumps(torch.randn(3, device="cpu")),
            )

            if HAS_MULTIGPU:
                self.assertEqual(
                    FxGraphCachePickler.dumps(torch.randn(3, device=f"{GPU_TYPE}:1")),
                    FxGraphCachePickler.dumps(torch.randn(3, device=f"{GPU_TYPE}:1")),
                )
                self.assertNotEqual(
                    FxGraphCachePickler.dumps(torch.randn(3, device=f"{GPU_TYPE}:0")),
                    FxGraphCachePickler.dumps(torch.randn(3, device=f"{GPU_TYPE}:1")),
                )

    def test_hash_kwargs(self):
        """
        Test the special handling of the kwargs when hashing, i.e.,
        ordering of the kwargs dict and any set arguments.
        """
        # Dict order of the kwargs should not affect hashes.
        details1 = FxGraphHashDetails(None, [], {"a": 0, "z": 1}, [])
        details2 = FxGraphHashDetails(None, [], {"z": 1, "a": 0}, [])
        self.assertEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )

        # Different kwarg values should affect hashes.
        details1 = FxGraphHashDetails(None, [], {"a": 0}, [])
        details2 = FxGraphHashDetails(None, [], {"a": 1}, [])
        self.assertNotEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )

        # Set order should not affect hashes. Sets are unordered, but
        # sorting and creating a new set seems to change the order.
        set1 = {"a", "b", "c", "d", "e", "f", "g"}
        set2 = set(sorted(set1))  # noqa: C414
        details1 = FxGraphHashDetails(None, [], {"a": set1}, [])
        details2 = FxGraphHashDetails(None, [], {"a": set2}, [])
        self.assertEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )

        # But different set contents should affect hashes.
        details1 = FxGraphHashDetails(None, [], {"a": {1, 2, 3}}, [])
        details2 = FxGraphHashDetails(None, [], {"a": {1, 2}}, [])
        self.assertNotEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )

    def test_hash_config_changes(self):
        """
        Test that different config settings affect hashes.
        """
        with config.patch({"max_autotune": False}):
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

        with config.patch({"max_autotune": True}):
            details3 = FxGraphHashDetails(None, [], {}, [])

        self.assertEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )
        self.assertNotEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details3),
        )

    def test_hash_custom_passes(self):
        """
        Test CustomGraphPass usage.
        """

        class TestCustomGraphPass(CustomGraphPass):
            def __init__(self):
                self._uuid = None

            def __call__(self, graph: torch.fx.graph.Graph) -> None:
                return None

            def uuid(self) -> Optional[Union[bytes, str]]:
                return self._uuid

        custom_pass = TestCustomGraphPass()
        with config.patch({"post_grad_custom_pre_pass": custom_pass}):
            custom_pass._uuid = "1"
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

            custom_pass._uuid = "2"
            details3 = FxGraphHashDetails(None, [], {}, [])

            self.assertEqual(
                FxGraphCachePickler.dumps(details1),
                FxGraphCachePickler.dumps(details2),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(details1),
                FxGraphCachePickler.dumps(details3),
            )

    def test_get_hash_for_files(self):
        """
        Test the get_hash_for_files helper.
        """
        with tempfile.NamedTemporaryFile(delete=True) as temp:
            temp.write(b"contents")
            temp.flush()

            hash1 = get_hash_for_files((temp.name,))
            get_hash_for_files.cache_clear()
            hash2 = get_hash_for_files((temp.name,))

            temp.write(b" ")
            temp.flush()
            get_hash_for_files.cache_clear()
            hash3 = get_hash_for_files((temp.name,))

            self.assertEqual(hash1, hash2)
            self.assertNotEqual(hash1, hash3)


class TestCudaCompileCommand(TestCase):
    @unittest.skipIf(not HAS_CUDA, "Requires CUDA")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_cuda_compile_command(self):
        cmd_no_extra_args: str = cuda_compile_command(
            ["abc.cu", "def.cu"], "output", "so"
        )
        assert "nvcc " in cmd_no_extra_args, cmd_no_extra_args
        assert "abc.cu" in cmd_no_extra_args, cmd_no_extra_args
        assert "def.cu" in cmd_no_extra_args, cmd_no_extra_args
        assert "output" in cmd_no_extra_args, cmd_no_extra_args
        cmd_extra_args: str = cuda_compile_command(
            ["abc.cu", "def.cu"], "output", "so", ["-Wwhatever", "-nothing"]
        )
        assert "nvcc " in cmd_extra_args, cmd_extra_args
        assert " -Wwhatever" in cmd_extra_args, cmd_extra_args
        assert " -nothing" in cmd_extra_args, cmd_extra_args
        assert "abc.cu" in cmd_extra_args, cmd_extra_args
        assert "def.cu" in cmd_extra_args, cmd_extra_args
        assert "output " in cmd_extra_args, cmd_extra_args
        with mock.patch("subprocess.check_output") as check_output_mock:
            CUDACodeCache.compile("test123.cu", "so", ["-Wsomething"])
            check_output_mock.assert_called()
            cmd_parts: List[str] = check_output_mock.call_args[0][0]
            assert cmd_parts[0] == "nvcc", cmd_parts
            assert "-Wsomething" in cmd_parts, cmd_parts
            assert "-DNDEBUG" in cmd_parts, cmd_parts


@instantiate_parametrized_tests
class TestAutotuneCache(TestCase):
    device_type = GPU_TYPE

    def setUp(self):
        super().setUp()
        counters.clear()
        PatchCaches.setUp()

    def tearDown(self):
        super().tearDown()
        PatchCaches.tearDown()

    def reset(self):
        torch._dynamo.reset()
        clear_inductor_caches()

    @unittest.skipIf(not HAS_CUDA, "Requires CUDA")
    @unittest.skipIf(not SM80OrLater, "Requires SM80+")
    @config.patch({"fx_graph_cache": False})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"autotune_local_cache": False})
    @config.patch({"autotune_remote_cache": True})
    @config.patch({"bundled_autotune_remote_cache": False})
    @config.patch({"max_autotune": True})
    def test_autotune_cache(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, a, b):
                return x + y, a + b

        def f(x, y, a, b):
            return Model()(x, y, a, b)

        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        a = torch.randn(1000, 100).cuda()
        b = torch.randn(1000, 100).cuda()
        f_compiled = torch.compile(f, fullgraph=True)

        with PatchCaches():
            f_compiled(x, y, a, b)

            self.assertEqual(global_stats.autotune_remote, Stats(2, 0, 2))

            self.reset()
            f_compiled(x, y, a, b)

        self.assertEqual(global_stats.autotune_remote, Stats(2, 2, 2))

        if config.is_fbcode():
            # Check that the cache entries seem reasonable
            for k in global_stats.autotune_remote.cache.keys():
                self.assertRegex(k, r"[0-9a-z]{52}\.py")
            for k in global_stats.triton.cache.keys():
                self.assertRegex(k, r"triton:[0-9a-f]{64}::[0-9a-f]{64}:c10")

    @unittest.skipIf(not HAS_CUDA, "Requires CUDA")
    @unittest.skipIf(not SM80OrLater, "Requires SM80+")
    @config.patch({"fx_graph_cache": False})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"autotune_local_cache": True})
    @config.patch({"autotune_remote_cache": False})
    @config.patch({"bundled_autotune_remote_cache": True})
    @config.patch({"max_autotune": True})
    def test_bundled_autotune_remote_cache(self):
        class Model(torch.nn.Module):
            def forward(self, a, b, c, d, e, f):
                return a + b, c + d, e + f

        def f(a, b, c, d, e, f):
            return Model()(a, b, c, d, e, f)

        f_compiled = torch.compile(f, fullgraph=True)

        a = torch.randn(101, 100).cuda()
        b = torch.randn(101, 100).cuda()
        c = torch.randn(102, 100).cuda()
        d = torch.randn(102, 100).cuda()
        e = torch.randn(103, 100).cuda()
        f = torch.randn(103, 100).cuda()

        with PatchCaches():
            f_compiled(a, b, c, d, e, f)

            self.assertEqual(global_stats.autotune_local, Stats(3, 0, 3))
            self.assertEqual(global_stats.bundled_autotune, Stats(1, 0, 1))

            self.reset()
            f_compiled(a, b, c, d, e, f)

        self.assertEqual(global_stats.autotune_local, Stats(6, 3, 3))
        self.assertEqual(global_stats.bundled_autotune, Stats(1, 1, 1))

        if config.is_fbcode():
            # Check that the cache entries seem reasonable
            for k in global_stats.autotune_local.cache.keys():
                self.assertRegex(k, r"tmp[^/]*/([^/]{2})/c\1[^/]{49}\.best_config")
            for k in global_stats.bundled_autotune.cache.keys():
                self.assertRegex(k, r"pt2:bundled-autotune-v1::[0-9a-z]{64}:c10")
            for k in global_stats.triton.cache.keys():
                self.assertRegex(k, r"triton:[0-9a-f]{64}::[0-9a-f]{64}:c10")


class TestRemoteAOTAutogradCache(TestCase):
    @unittest.skipIf(not HAS_CUDA, "Requires CUDA")
    @unittest.skipIf(not SM80OrLater, "Requires SM80+")
    @config.patch({"fx_graph_cache": False})
    @config.patch({"fx_graph_remote_cache": True})
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @torch._functorch.config.patch({"enable_remote_autograd_cache": True})
    def test_autograd_remote_cache(self):
        def f(a, b):
            return a + b

        f_compiled = torch.compile(f)
        a = torch.randn(101, 100, device="cuda", requires_grad=False)
        b = torch.randn(101, 100, device="cuda", requires_grad=False)
        with PatchCaches():
            f_compiled(a, b)

            self.assertEqual(global_stats.aot_autograd, Stats(1, 0, 1))
            self.assertEqual(global_stats.fx_graph, Stats(1, 0, 1))

            torch._dynamo.reset()

            f_compiled(a, b)
            self.assertEqual(global_stats.aot_autograd, Stats(1, 1, 1))
            self.assertEqual(global_stats.fx_graph, Stats(1, 1, 1))

        if config.is_fbcode():
            # Check that the cache entries seem reasonable
            for k in global_stats.aot_autograd.cache.keys():
                self.assertRegex(k, r"pt2:autograd-experimental::[0-9a-z]{52}:c10")

            for k in global_stats.fx_graph.cache.keys():
                self.assertRegex(k, r"pt2:fx-graph-v1::[0-9a-z]{52}:c10")

    @unittest.skipIf(not HAS_CUDA, "Requires CUDA")
    @unittest.skipIf(not SM80OrLater, "Requires SM80+")
    @config.patch({"fx_graph_cache": False})
    @config.patch({"fx_graph_remote_cache": True})
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @torch._functorch.config.patch({"enable_remote_autograd_cache": True})
    def test_autograd_remote_lazy_backward(self):
        """
        Lazily compile the backward, and lazily save to cache
        """

        def fn(a, b):
            return a.cos() + b

        with PatchCaches():
            a = torch.randn(25, requires_grad=True)
            b = torch.randn(25, requires_grad=True)
            a2 = a.detach().clone().requires_grad_(True)
            b2 = b.detach().clone().requires_grad_(True)
            compiled_fn = torch.compile(fn, backend="inductor")
            self.assertEqual(fn(a, b), compiled_fn(a2, b2))
            self.assertEqual(global_stats.aot_autograd, Stats(0, 0, 1))

            # Clear dynamo and run again. Should be a cache miss still, because backward hasn't run
            torch._dynamo.reset()
            self.assertEqual(fn(a, b), compiled_fn(a2, b2))
            self.assertEqual(global_stats.aot_autograd, Stats(0, 0, 2))

            # Now let's run the backward
            fn(a, b).sum().backward()
            compiled_fn(a2, b2).sum().backward()
            self.assertEqual(a.grad, a2.grad)
            self.assertEqual(b.grad, b2.grad)
            self.assertEqual(global_stats.aot_autograd, Stats(1, 0, 2))

            # Clear dynamo and rerun everything, now there should be a cache hit
            torch._dynamo.reset()
            a = torch.randn(25, requires_grad=True)
            b = torch.randn(25, requires_grad=True)
            a2 = a.detach().clone().requires_grad_(True)
            b2 = b.detach().clone().requires_grad_(True)
            self.assertEqual(fn(a, b), compiled_fn(a2, b2))
            self.assertEqual(global_stats.aot_autograd, Stats(1, 1, 2))

            fn(a, b).sum().backward()
            compiled_fn(a2, b2).sum().backward()
            self.assertEqual(a.grad, a2.grad)
            self.assertEqual(b.grad, b2.grad)


class TestUtils(TestCase):
    @config.patch({"fx_graph_remote_cache": False})
    def test_fresh_inductor_cache(self):
        def fn(x, y):
            return x + y

        a = torch.rand(10)
        b = torch.rand(10)

        with fresh_inductor_cache():
            self.assertEqual(len(PyCodeCache.cache.keys()), 0)
            res1 = torch.compile(fn)(a, b)
            cache_dir1 = cache_dir()

        torch._dynamo.reset()
        with fresh_inductor_cache():
            self.assertEqual(len(PyCodeCache.cache.keys()), 0)
            res2 = torch.compile(fn)(a, b)
            cache_dir2 = cache_dir()

        self.assertEqual(res1, res2)
        self.assertNotEqual(cache_dir1, cache_dir2)


if __name__ == "__main__":
    run_tests()
