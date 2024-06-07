# Owner(s): ["module: inductor"]
import base64
import functools
import json
import os
import pickle
import unittest
from typing import List
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
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import clear_inductor_caches, fresh_inductor_cache
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA,
    HAS_GPU,
    HAS_MULTIGPU,
)
from torch.utils._triton import has_triton

HAS_TRITON = has_triton()

if HAS_TRITON:
    import triton

    from torch.testing._internal.triton_utils import add_kernel

requires_gpu = functools.partial(unittest.skipIf, not HAS_GPU, "requires gpu")
requires_triton = functools.partial(unittest.skipIf, not HAS_TRITON, "requires triton")

torch._dynamo.config.fake_tensor_cache_enabled = True
torch._dynamo.config.fake_tensor_cache_crosscheck_enabled = True


class MyModel(torch.nn.Module):
    def __init__(self):
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
    def setUp(self):
        super().setUp()
        counters.clear()

    def reset(self):
        torch._dynamo.reset()
        clear_inductor_caches()

    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    def test_cache_load_function(self, device, dtype, dynamic):
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

        compiled_fn = torch.compile(fn, dynamic=dynamic)

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        for m in torch._inductor.codecache.PyCodeCache.cache.values():
            os.remove(m.__file__)
        self.reset()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @requires_triton()
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    def test_remote_cache_load_function(self, device, dtype, dynamic):
        from unittest.mock import patch

        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25, dtype=dtype, device=device)
        b = torch.rand(5, 5, dtype=dtype, device=device)

        cache = {}
        num_get = 0
        num_put = 0

        class MyCache:
            def __init__(self, key, is_autotune=False):
                pass

            def get(self, filename):
                nonlocal cache
                nonlocal num_get
                if filename not in cache:
                    return None
                ret = json.loads(cache[filename])
                num_get += 1
                if config.is_fbcode():
                    return base64.b64decode(ret["data"]) if ret is not None else ret
                else:
                    return base64.b64decode(ret) if ret is not None else ret

            def put(self, filename, data):
                nonlocal cache
                nonlocal num_put
                if config.is_fbcode():
                    data["data"] = base64.b64encode(data["data"]).decode("ascii")
                else:
                    data = base64.b64encode(data).decode("ascii")
                cache[filename] = json.dumps(data)
                num_put += 1

        cache_module = (
            "triton.runtime.fb_memcache.FbMemcacheRemoteFxGraphCacheBackend"
            if config.is_fbcode()
            else "torch._inductor.remote_cache.RedisRemoteCacheBackend"
        )

        with config.patch(
            {
                "fx_graph_cache": False,
                "fx_graph_remote_cache": True,
            }
        ), patch.dict(os.environ), patch(cache_module, MyCache, create=True):
            os.environ.pop("TRITON_CACHE_MANAGER", None)
            for _ in range(4):
                with fresh_inductor_cache():
                    compiled_fn = torch.compile(fn, dynamic=dynamic)
                    self.assertEqual(fn(a, b), compiled_fn(a, b))
                reset()
            self.assertEqual(num_get, 3)
            self.assertEqual(num_put, 1)

    @requires_triton()
    @config.patch({"fx_graph_cache": True})
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

    @requires_gpu()
    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    def test_higher_order_op_bypass(self):
        """
        Verify that we bypass the cache when we have higher order ops.
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

    @skipIfRocm
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


class TestUtils(TestCase):
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
