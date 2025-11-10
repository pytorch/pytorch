# Owner(s): ["module: inductor"]
import functools
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import contextmanager
from typing import Optional, Union
from typing_extensions import override
from unittest import mock

import torch
from torch._dynamo import reset
from torch._dynamo.package import DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch._inductor import config, metrics
from torch._inductor.codecache import (
    BypassFxGraphCache,
    cuda_compile_command,
    CUDACodeCache,
    FxGraphCachePickler,
    FxGraphHashDetails,
    PyCodeCache,
    TensorMetadata,
    TensorMetadataAndValues,
)
from torch._inductor.cpp_builder import normalize_path_separator
from torch._inductor.custom_graph_pass import (
    CustomGraphModulePass,
    CustomGraphPass,
    CustomPartitionerFn,
    get_hash_for_files,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.mock_cache import global_stats, PatchCaches, Stats
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import clear_caches, fresh_cache
from torch._library import capture_triton
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
)
from torch.testing._internal.common_cuda import (
    SM80OrLater,
    TEST_MULTIGPU,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_SANDCASTLE,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    HAS_MULTIGPU,
    HAS_TRITON,
    HAS_XPU_AND_TRITON,
    patch_inductor_backend,
    requires_gpu,
    requires_triton,
)
from torch.testing._internal.triton_utils import (
    requires_cuda_and_triton,
    requires_gpu_and_triton,
)


try:
    from . import custom_inductor_config
except ImportError:
    import custom_inductor_config


if HAS_TRITON:
    import triton  # @manual

    from torch.testing._internal.triton_utils import add_kernel, sub_kernel

torch._dynamo.config.fake_tensor_cache_enabled = True
torch._dynamo.config.fake_tensor_cache_crosscheck_enabled = True


class LogCaptureHandler(logging.Handler):
    def __init__(self, level):
        super().__init__(level)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextmanager
def capture_logs(log_name, log_level):
    try:
        logger = logging.getLogger(log_name)
        old_level = logger.level
        handler = logging.Handler()
        logger.setLevel(log_level)
        log_records = []

        def emit(record):
            log_records.append(record)

        handler.emit = emit
        logger.addHandler(handler)

        yield log_records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


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


class TestPyCodeCache(TestCase):
    def test_linemaps_empty(self):
        src = """import torch"""
        (key, path) = PyCodeCache.write(src, "")
        # Load with an empty linemap
        PyCodeCache.load_by_key_path(key, path, linemap=[])
        stack_frames = PyCodeCache.stack_frames_for_code(path, 0)
        self.assertEqual(stack_frames, None)

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_editable_cached_wrapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["TORCHINDUCTOR_CACHE_DIR"] = tmpdir

            step1 = textwrap.dedent(
                """
                import glob
                import os
                import torch
                import warnings
                from torch._inductor import config

                warnings.filterwarnings("ignore")
                config.fx_graph_cache = True
                config.fx_graph_remote_cache = False
                torch._dynamo.reset()

                @torch.compile(backend="inductor")
                def f(x):
                    return x * 2

                f(torch.ones(2))
                cache_dir = os.environ["TORCHINDUCTOR_CACHE_DIR"]
                pyfiles = glob.glob(os.path.join(cache_dir, "**", "*.py"), recursive=True)
                print(pyfiles[0])
                """
            )
            wrapper_path = (
                subprocess.check_output([sys.executable, "-c", step1], env=env)
                .decode()
                .strip()
            )

            step2 = textwrap.dedent(
                """
                import torch
                import warnings
                from torch._dynamo.utils import counters
                from torch._inductor import config

                warnings.filterwarnings("ignore")
                config.fx_graph_cache = True
                config.fx_graph_remote_cache = False
                torch._dynamo.reset()

                @torch.compile(backend="inductor")
                def f(x):
                    return x * 2

                f(torch.ones(2))
                print(counters["inductor"]["fxgraph_cache_hit"])
                """
            )
            hit = (
                subprocess.check_output([sys.executable, "-c", step2], env=env)
                .decode()
                .strip()
            )
            # XPU have extra lines, so get the last line, refer https://github.com/intel/torch-xpu-ops/issues/2261
            if torch.xpu.is_available():
                wrapper_path = wrapper_path.splitlines()[-1]
                hit = hit.splitlines()[-1]
            self.assertEqual(hit, "1")

            with open(wrapper_path) as f:
                src = f.read()
            with open(wrapper_path, "w") as f:
                f.write(
                    src.replace(
                        "def call(self, args):",
                        "def call(self, args):\n        print('debug')",
                    )
                )

            step3 = textwrap.dedent(
                """
                import torch
                import warnings
                from torch._inductor import config

                warnings.filterwarnings("ignore")
                config.fx_graph_cache = True
                config.fx_graph_remote_cache = False
                torch._dynamo.reset()

                @torch.compile(backend="inductor")
                def f(x):
                    return x * 2

                f(torch.ones(2))
                """
            )
            out = subprocess.check_output(
                [sys.executable, "-c", step3], env=env
            ).decode()
            self.assertIn("debug", out)


@instantiate_parametrized_tests
class TestFxGraphCache(TestCase):
    device_type = GPU_TYPE

    def setUp(self):
        super().setUp()
        counters.clear()
        DynamoCache.clear()
        PrecompileContext.clear()
        AOTAutogradCache.clear()
        PatchCaches.setUp()
        CacheArtifactManager.clear()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        PatchCaches.tearDown()

    def reset(self):
        AOTAutogradCache.clear()
        DynamoCache.clear()
        PrecompileContext.clear()
        PyCodeCache.cache_clear(purge=True)
        torch._dynamo.reset()
        clear_caches()

    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"compile_threads": 1})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    @parametrize("bundle_triton", (False, True))
    @parametrize("use_static_cuda_launcher", (False, True))
    @parametrize("grad", (False, True))
    def test_cache_load_function(
        self, device, dtype, dynamic, bundle_triton, use_static_cuda_launcher, grad
    ):
        """
        Verify that we can populate and load functions from the cache.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")
        if use_static_cuda_launcher and not (device == "cuda" and bundle_triton):
            raise unittest.SkipTest(
                "Static cuda launcher requires cuda and triton bundling"
            )
        if use_static_cuda_launcher and TEST_WITH_ROCM:
            raise unittest.SkipTest("Static cuda launcher doesn't work with ROCM")

        grad_multiplier = 2 if grad else 1

        def fn(x, y):
            yy = y @ y
            return x * 2 + yy.view(25)

        a_orig = torch.rand(25, dtype=dtype, device=device)
        b_orig = torch.rand(5, 5, dtype=dtype, device=device)

        with config.patch(
            bundle_triton_into_fx_graph_cache=bundle_triton,
            use_static_cuda_launcher=use_static_cuda_launcher,
        ):
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            a1 = a_orig.clone().requires_grad_(grad)
            b1 = b_orig.clone().requires_grad_(grad)
            a2 = a_orig.clone().requires_grad_(grad)
            b2 = b_orig.clone().requires_grad_(grad)

            # A first call should miss in the cache.
            eager_result = fn(a1, b1)
            compiled_result = compiled_fn(a2, b2)
            self.assertEqual(eager_result, compiled_result)
            if grad:
                eager_result.sum().backward()
                compiled_result.sum().backward()
                self.assertEqual(a1.grad, a2.grad)
                self.assertEqual(b1.grad, b2.grad)
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_miss"], grad_multiplier * 1
            )
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

            # we expect:
            #  .ttir
            #  .ttgir
            #  .llir
            #  .ptx (cuda) or .spv (xpu)
            #  .json
            #  __grp__.*.json
            # optionally, we can also get
            #  .cubin (CUDA only)
            #  .source (new versions of triton only, triton-lang/triton#6992)

            # to avoid depending on the device and triton version, just assert that
            # we have at least 6 kernels.
            save_and_read_min_artifact_count = 6
            if bundle_triton and device != "cpu":
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_save_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count,
                )
                self.assertEqual(
                    counters["inductor"]["triton_bundler_read_and_emit_kernel"], 0
                )
                if use_static_cuda_launcher:
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_save_static_autotuner"],
                        grad_multiplier if device == "cuda" else 0,
                    )
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_load_static_autotuner"], 0
                    )

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()

            # Clean triton kernels
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

            a1 = a_orig.clone().requires_grad_(grad)
            b1 = b_orig.clone().requires_grad_(grad)
            a2 = a_orig.clone().requires_grad_(grad)
            b2 = b_orig.clone().requires_grad_(grad)

            eager_result = fn(a1, b1)
            compiled_result = compiled_fn(a2, b2)
            self.assertEqual(eager_result, compiled_result)
            if grad:
                eager_result.sum().backward()
                compiled_result.sum().backward()
                self.assertEqual(a1.grad, a2.grad)
                self.assertEqual(b1.grad, b2.grad)
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_miss"], grad_multiplier * 1
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_hit"], grad_multiplier * 1
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_lookup_write_file"], grad_multiplier * 1
            )

            if bundle_triton and device != "cpu":
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_save_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count,
                )
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_read_and_emit_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count,
                )
                if use_static_cuda_launcher:
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_save_static_autotuner"],
                        grad_multiplier if device == "cuda" else 0,
                    )
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_load_static_autotuner"],
                        grad_multiplier if device == "cuda" else 0,
                    )

            self.reset()

            a1 = a_orig.clone().requires_grad_(grad)
            b1 = b_orig.clone().requires_grad_(grad)
            a2 = a_orig.clone().requires_grad_(grad)
            b2 = b_orig.clone().requires_grad_(grad)

            eager_result = fn(a1, b1)
            if grad:
                eager_result.sum().backward()
            with torch.compiler.config.patch({"cache_key_tag": "test"}):
                compiled_result = compiled_fn(a2, b2)
                if grad:
                    compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            if grad:
                self.assertEqual(a1.grad, a2.grad)
                self.assertEqual(b1.grad, b2.grad)

            self.assertEqual(
                counters["inductor"]["fxgraph_cache_miss"], grad_multiplier * 2
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_hit"], grad_multiplier * 1
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_lookup_write_file"], grad_multiplier * 1
            )

            if bundle_triton and device != "cpu":
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_save_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count * 2,
                )
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_read_and_emit_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count,
                )
                if use_static_cuda_launcher:
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_save_static_autotuner"],
                        grad_multiplier * 2 if device == "cuda" else 0,
                    )
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_load_static_autotuner"],
                        grad_multiplier if device == "cuda" else 0,
                    )

    @requires_triton()
    @config.patch({"fx_graph_remote_cache": True})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    @parametrize("bundle_triton", (False, True))
    @parametrize("use_static_cuda_launcher", (False, True))
    @config.patch(
        {"compile_threads": 1}
    )  # Can't check globalStats if there are workers
    def test_remote_cache_load_function(
        self, device, dtype, dynamic, bundle_triton, use_static_cuda_launcher
    ):
        from unittest.mock import patch

        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")
        if use_static_cuda_launcher and not (device == "cuda" and bundle_triton):
            raise unittest.SkipTest(
                "Static cuda launcher requires cuda and triton bundling"
            )
        if use_static_cuda_launcher and TEST_WITH_ROCM:
            raise unittest.SkipTest("Static cuda launcher doesn't work with ROCM")

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25, dtype=dtype, device=device)
        b = torch.rand(5, 5, dtype=dtype, device=device)

        with (
            config.patch(
                {
                    "fx_graph_remote_cache": True,
                    "bundle_triton_into_fx_graph_cache": bundle_triton,
                    "use_static_cuda_launcher": use_static_cuda_launcher,
                }
            ),
            patch.dict(os.environ),
            PatchCaches(),
        ):
            os.environ.pop("TRITON_CACHE_MANAGER", None)
            for _ in range(4):
                with fresh_cache():
                    compiled_fn = torch.compile(fn, dynamic=dynamic)
                    self.assertEqual(fn(a, b), compiled_fn(a, b))
                reset()

            self.assertEqual(global_stats.fx_graph, Stats(1, 3, 1))

            with torch.compiler.config.patch({"cache_key_tag": "test"}), fresh_cache():
                compiled_fn = torch.compile(fn, dynamic=dynamic)
                self.assertEqual(fn(a, b), compiled_fn(a, b))

            self.assertEqual(global_stats.fx_graph, Stats(2, 3, 2))

        # Check that the cache entries seem reasonable
        for k in global_stats.fx_graph.cache.keys():
            self.assertRegex(k, r"pt2:fx-graph-v1::[0-9a-z]{52}:c[0-9]+")

    @requires_triton()
    @config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "autotune_local_cache": True,
        }
    )
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    def test_cache_hot_load(self, device, dtype, dynamic):
        """
        Verify that we can populate and hot load functions from the cache.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        def fn(x, y):
            return x.sin() @ y

        a = torch.rand(100, 100, dtype=dtype, device=device)
        b = torch.rand(100, 100, dtype=dtype, device=device)

        # Record artifacts
        with fresh_cache():
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            # A first call should miss in the cache.
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        autotune_expect = 1 if device == GPU_TYPE else 0

        self.assertEqual(len(cache_info.inductor_artifacts), 1)
        self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 0)
        self.assertEqual(len(cache_info.pgo_artifacts), 0)

        self.reset()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # We did not load anything so dont hit yet
        with fresh_cache():
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        self.reset()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with fresh_cache():
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)

            self.assertEqual(len(cache_info.inductor_artifacts), 1)
            self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 0)
            self.assertEqual(len(cache_info.pgo_artifacts), 0)

            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @requires_triton()
    @config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "autotune_local_cache": True,
        }
    )
    @torch._dynamo.config.patch(
        {
            "caching_precompile": True,
        }
    )
    @parametrize("dynamic", (False, True))
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    def test_cache_hot_load_caching_precompile(self, device, dtype, dynamic):
        """
        Verify that we can populate and hot load functions from the cache.
        """

        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        def fn(x, y):
            return x.sin() @ y

        a = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)
        b = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)

        # Record artifacts
        with fresh_cache():
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            # A first call should miss in the cache.
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_miss"], 1)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        autotune_expect = 2 if device == GPU_TYPE else 0
        self.assertEqual(len(cache_info.inductor_artifacts), 2)
        self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
        self.assertEqual(len(cache_info.pgo_artifacts), 0)
        self.assertEqual(len(cache_info.precompile_artifacts), 1)

        self.reset()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # We did not load anything so dont hit yet
        with fresh_cache():
            eager_result = fn(a, b)
            # With caching precompile, we have to re torch.compile the function
            # to trigger cache lookup
            compiled_fn = torch.compile(fn, dynamic=dynamic)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_miss"], 2)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 0)
        self.reset()
        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with fresh_cache(), torch.compiler.set_stance("fail_on_recompile"):
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
            self.assertEqual(len(cache_info.inductor_artifacts), 2)
            self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
            self.assertEqual(len(cache_info.pgo_artifacts), 0)
            self.assertEqual(len(cache_info.precompile_artifacts), 1)

            # With caching precompile, we have to re torch.compile the function
            # to trigger cache lookup
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_miss"], 2)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 1)

    @config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
        }
    )
    def test_cache_hot_load_repeat(self):
        def fn(x, y):
            return x @ y.sin()

        compiled_fn = torch.compile(fn, dynamic=False)

        a = torch.randn(4, 4)
        b = torch.randn(4, 4)

        a2 = torch.randn(4, 8)
        b2 = torch.randn(8, 4)

        with fresh_cache():
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertFalse(torch.compiler._cache.CacheArtifactManager.need_serialize())
        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        self.reset()

        with fresh_cache():
            torch.compiler.load_cache_artifacts(artifact_bytes)
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        self.assertFalse(torch.compiler._cache.CacheArtifactManager.need_serialize())

        self.reset()

        with fresh_cache():
            eager_result = fn(a2, b2)
            compiled_result = compiled_fn(a2, b2)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        self.assertTrue(torch.compiler._cache.CacheArtifactManager.need_serialize())

    @torch._dynamo.config.patch(automatic_dynamic_local_pgo=True)
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @config.patch({"fx_graph_cache": True, "fx_graph_remote_cache": False})
    def test_cache_hot_load_pgo(self):
        """
        Verify that we can populate and hot load functions from the cache with pgo.
        """

        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        @torch.compile(backend=backend, fullgraph=True)
        def f(x):
            return x * 2

        # Record artifacts
        with torch.compiler.config.patch(job_id=self.id()), fresh_cache():
            f(torch.randn(2, 3))
            f(torch.randn(2, 4))
            self.assertEqual(backend.frame_count, 2)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        self.assertEqual(len(cache_info.inductor_artifacts), 2)
        self.assertEqual(len(cache_info.autotune_artifacts), 0)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 0)
        self.assertEqual(len(cache_info.pgo_artifacts), 2)

        self.reset()
        backend.clear()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with torch.compiler.config.patch({"job_id": self.id()}), fresh_cache():
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)

            self.assertEqual(len(cache_info.inductor_artifacts), 2)
            self.assertEqual(len(cache_info.autotune_artifacts), 0)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 0)
            self.assertEqual(len(cache_info.pgo_artifacts), 2)

            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
            self.assertEqual(backend.frame_count, 1)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @torch._dynamo.config.patch(automatic_dynamic_local_pgo=True)
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @config.patch({"fx_graph_cache": True, "fx_graph_remote_cache": False})
    def test_cache_hot_load_pgo_swap_file_names(self):
        """
        Verify that we can populate and hot load functions from the cache with pgo
        with file name swapping
        """

        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        @torch.compile(backend=backend, fullgraph=True)
        def f(x):
            return x * 2

        # Record artifacts
        with mock.patch(
            "torch._utils_internal.get_mast_job_name_version", return_value=("foo", 5)
        ):
            with fresh_cache():
                f(torch.randn(2, 3))
                f(torch.randn(2, 4))
                self.assertEqual(backend.frame_count, 2)

            artifacts = torch.compiler.save_cache_artifacts()

            self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        self.assertEqual(len(cache_info.pgo_artifacts), 2)

        self.reset()
        backend.clear()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with (
            mock.patch(
                "torch._utils_internal.get_mast_job_name_version",
                return_value=("bar", 10),
            ),
            fresh_cache(),
        ):
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)

            self.assertEqual(len(cache_info.pgo_artifacts), 2)

            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
            self.assertEqual(backend.frame_count, 1)

    def test_cache_hot_load_empty(self):
        self.assertIsNone(torch.compiler.save_cache_artifacts())

    def test_cache_hot_load_generic(self):
        class CacheStub:
            def __init__(self):
                self.cache = {}

            def lookup(self, key):
                content = self.cache.get(key)
                if content is None:
                    return None

                CacheArtifactManager.record_artifact(
                    ArbitraryCacheArtifact.type(), key, content
                )
                return content

            def save(self, key, content):
                self.cache[key] = content
                CacheArtifactManager.record_artifact(
                    ArbitraryCacheArtifact.type(), key, content
                )

            def clear(self):
                self.cache.clear()

        cache_stub = CacheStub()

        @CacheArtifactFactory.register
        class ArbitraryCacheArtifact(CacheArtifact):
            @override
            def populate_cache(self) -> None:
                cache_stub.cache[self.key] = self.content.decode()

            @override
            @staticmethod
            def type() -> str:
                return "test"

            @override
            @staticmethod
            def encode(content: str) -> bytes:
                return content.encode()

        test_cache = {"1": "foo", "2": "bar", "foo": "bar"}

        for k, v in test_cache.items():
            cache_stub.save(k, v)

        artifacts = torch.compiler.save_cache_artifacts()
        self.assertIsNotNone(artifacts)
        artifact_bytes, cache_info = artifacts

        self.assertEqual(len(cache_info.test_artifacts), 3)

        cache_stub.clear()
        CacheArtifactManager.clear()

        cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
        self.assertEqual(len(cache_info.test_artifacts), 3)
        self.assertEqual(cache_stub.cache, test_cache)

        CacheArtifactManager.clear()
        cache_stub.lookup("foo")
        artifacts = torch.compiler.save_cache_artifacts()
        self.assertIsNotNone(artifacts)
        _, cache_info = artifacts
        self.assertEqual(len(cache_info.test_artifacts), 1)

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
        inp = torch.randn(2, 3, 16, 32, device=device, dtype=dtype)

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

    @largeTensorTest("64GB", device=GPU_TYPE, inductor=True)
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

    @config.patch("fx_graph_cache", True)
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @config.patch("fx_graph_remote_cache", False)
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @requires_cuda_and_triton
    def test_no_arguments_tensor_device_guards(self):
        """
        Usually, when there are example inputs, the device index of the inputs
        is sufficient to make sure we don't cache hit with the results from different
        cuda devices.
        When the input has no arguments, we still need to have the cuda
        device index in the cache key.
        """

        @torch.compile
        def f():
            y = torch.randn(3, device="cuda")
            return (y,)

        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            result = f()
            self.assertEqual(result[0].device, torch.device("cuda:0"))
        self.reset()
        # Should not cache hit with device guard
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            result = f()
            self.assertEqual(result[0].device, torch.device("cuda:1"))

    @config.patch("fx_graph_cache", True)
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @config.patch("fx_graph_remote_cache", False)
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @requires_cuda_and_triton
    def test_tensor_device_guards_cpu_tensor(self):
        """
        CPU tensor arguments should still cache hit
        """

        @torch.compile
        def f(x):
            return x.sin()

        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            result = f(torch.randn(3, device="cpu"))
            self.assertEqual(result.device, torch.device("cpu"))

        self.reset()
        # Should not cache hit with device guard
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            result = f(torch.randn(3, device="cpu"))
            self.assertEqual(result.device, torch.device("cpu"))

        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

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
            return x + torch.tensor(list(range(12)), device=device)

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

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("variant", ("v1", "v2"))
    def test_auto_functionalized_caching(self, variant):
        if variant == "v1":
            patch = torch._inductor.config.patch(enable_auto_functionalized_v2=False)
        else:
            assert variant == "v2"
            patch = torch._inductor.config.patch(enable_auto_functionalized_v2=True)

        @torch.library.custom_op("mylib::sin_inplace", mutates_args=["x"])
        def sin_inplace(x: torch.Tensor) -> None:
            x.sin_()

        @torch.library.custom_op("mylib::cos_inplace", mutates_args=["x"])
        def cos_inplace(x: torch.Tensor) -> None:
            x.cos_()

        @torch.compile(fullgraph=True)
        def fn(x, op):
            y = torch.empty_like(x)
            op(y)
            return y

        x = torch.randn(3)

        with patch:
            # A first call should miss in the cache.
            fn(x, sin_inplace)
            self.reset()
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()
            fn(x, sin_inplace)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

            # A third call with different operator should have a cache miss
            self.reset()
            fn(x, cos_inplace)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @requires_gpu_and_triton
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @with_tf32_off
    def test_flex_attention_caching(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        block_mask = create_block_mask(
            lambda b, h, q, kv: q >= kv, None, None, 512, 512
        )

        def score_mod(score, b, h, q, kv):
            return score + (q - kv)

        def fn(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

        def score_mod2(score, b, h, q, kv):
            return score

        def fn2(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod2, block_mask=block_mask)

        a, b, c = (torch.randn(1, 4, 512, 64).to(GPU_TYPE) for _ in range(3))
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
        self.reset()
        self.assertEqual(fn(a, b, c), compiled_fn(a, b, c), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

        # A third call with different score_mod should have a cache miss
        self.reset()
        self.assertEqual(fn2(a, b, c), compiled_fn2(a, b, c), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @requires_gpu()
    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("bundle_triton", (False, True))
    def test_higher_order_op_bypass(self, bundle_triton):
        """
        Verify that we bypass the cache when we have a higher order ops
        and that bundler start/end works with a cache bypass.
        """

        def fn(x):
            def true_fn(x: torch.Tensor):
                return x.cos()

            def false_fn(x: torch.Tensor):
                return x.sin()

            return torch.cond(x.shape[0], true_fn, false_fn, (x,))

        with config.patch(
            bundle_triton_into_fx_graph_cache=bundle_triton,
        ):
            compiled_fn = torch.compile(fn, dynamic=True, fullgraph=True)

            x = torch.randn(4, 4, device=GPU_TYPE)
            compiled_fn(x)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertGreater(counters["inductor"]["fxgraph_cache_bypass"], 0)

    @requires_gpu()
    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("bundle_triton", (False, True))
    def test_triton_higher_order_op(self, bundle_triton):
        """
        Verify that we can cache user defined triton kernel higher order op
        """

        def fn(x, y):
            n_elements = x.numel()
            grid = lambda meta: (  # noqa: E731
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
            )
            add_kernel[grid](x, y, x, n_elements, BLOCK_SIZE=4)
            return x

        def fn2(x, y):
            n_elements = x.numel()
            grid = lambda meta: (  # noqa: E731
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
            )
            sub_kernel[grid](x, y, x, n_elements, BLOCK_SIZE=4)
            return x

        with config.patch(bundle_triton_into_fx_graph_cache=bundle_triton):
            compiled_fn = torch.compile(fn, fullgraph=True)
            compiled_fn2 = torch.compile(fn2, fullgraph=True)

            x = torch.randn(4, device=GPU_TYPE)
            y = torch.randn(4, device=GPU_TYPE)

            compiled_fn(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()

            # Clean PyCodeCache and triton kernels
            PyCodeCache.cache_clear()
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

            compiled_fn(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()

            # Clean PyCodeCache and triton kernels
            PyCodeCache.cache_clear()
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

            compiled_fn2(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

    @requires_gpu()
    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("bundle_triton", (False, True))
    def test_triton_higher_order_op_different_configs(self, bundle_triton):
        """
        Verify that user defined triton kernel with
        different configs are cached separately.
        """

        add_kernel1 = triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=8),
                triton.Config({"BLOCK_SIZE": 128}, num_stages=4, num_warps=4),
            ],
            key=[],
        )(add_kernel)

        add_kernel2 = triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=8),
                triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
            ],
            key=[],
        )(add_kernel)

        def fn(x, y):
            n_elements = x.numel()
            grid = lambda meta: (  # noqa: E731
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
            )
            add_kernel1[grid](x, y, x, n_elements)
            return x

        def fn2(x, y):
            n_elements = x.numel()
            grid = lambda meta: (  # noqa: E731
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
            )
            add_kernel2[grid](x, y, x, n_elements)
            return x

        with config.patch(bundle_triton_into_fx_graph_cache=bundle_triton):
            compiled_fn = torch.compile(fn, fullgraph=True)
            compiled_fn2 = torch.compile(fn2, fullgraph=True)

            x = torch.randn(4, device=GPU_TYPE)
            y = torch.randn(4, device=GPU_TYPE)

            compiled_fn(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()

            # Clean PyCodeCache and triton kernels
            PyCodeCache.cache_clear()
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

            compiled_fn(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()

            # Clean PyCodeCache and triton kernels
            PyCodeCache.cache_clear()
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

            compiled_fn2(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

    @requires_gpu()
    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"compile_threads": 1})
    @parametrize("bundle_triton", (False, True))
    @parametrize("use_static_cuda_launcher", (False, True))
    def test_triton_op(self, bundle_triton, use_static_cuda_launcher):
        if use_static_cuda_launcher and TEST_WITH_ROCM:
            raise unittest.SkipTest("Static cuda launcher doesn't work with ROCM")

        libname = "my_cool_namespace"
        opname = "my_triton_operator"

        @torch._library.triton_op(f"{libname}::{opname}", mutates_args={})
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
            return output

        def f(x, y):
            return add(x, y)

        compile_threads = 1 if use_static_cuda_launcher else config.compile_threads
        with config.patch(
            bundle_triton_into_fx_graph_cache=bundle_triton,
            use_static_cuda_launcher=use_static_cuda_launcher,
            compile_threads=compile_threads,
        ):
            compiled_fn = torch.compile(f, fullgraph=True)

            x = torch.randn(4, device=GPU_TYPE)
            y = torch.randn(4, device=GPU_TYPE)

            compiled_fn(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()

            # Clean PyCodeCache and triton kernels
            PyCodeCache.cache_clear()
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

            compiled_fn(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_generated_kernel_count(self):
        """
        Test that we bump the generated_kernel_count metric on a cache hit.
        """
        torch._logging.set_logs(inductor_metrics=True)

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
        torch._logging.set_logs()

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_inductor_counters(self):
        """
        Test that we bump the inductor counters on a cache hit.
        """

        def fn(a, b):
            return torch.mm(a, b)

        a = torch.rand(8, 32, device="cpu")
        b = torch.rand(32, 8, device="cpu")

        compiled_fn = torch.compile(fn)

        # Verify the "miss" case.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # Verify the "hit" case.
        self.reset()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

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

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    @requires_cuda_and_triton
    @unittest.expectedFailure  # TODO: pass in optimize_mem at runtime
    def test_async_compile_cache(self):
        class SimpleFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * 2

        x = torch.rand([10], requires_grad=True, device="cuda")
        counters.clear()

        sf = SimpleFunction
        out = torch.compile(sf.apply)(x)
        out.sum().backward()

        self.assertEqual(counters["inductor"]["async_compile_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["async_compile_cache_hit"], 1)

    @config.patch({"fx_graph_cache": True})
    def test_cache_guard_overspec(self):
        b = torch.tensor([0, 2, 4, 6, 8])

        @torch.compile
        class MyModel(torch.nn.Module):
            def forward(self, x):
                return torch.isin(x, b)

        model = MyModel()

        counters.clear()

        for i in range(1, 5):
            model(torch.arange(i))

        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        self.reset()
        counters.clear()

        for i in range(1, 5):
            model(torch.arange(i))

        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"freezing": True})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("inlinable", (True, False))
    def test_freezing(self, device, inlinable):
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        # For machines with mkldnn_fp16 support, weight_pack in mkldnn_fusion.py causes
        # the creation of a mkldnn format tensor which the current implementation does
        # not support.
        if (
            device == "cpu"
            and torch.backends.mkldnn.is_available()
            and torch.ops.mkldnn._is_mkldnn_fp16_supported()
        ):
            raise unittest.SkipTest("mkldnn tensors unsupported")

        # The shape of the frozen constant determines if it will be inlined.
        shape = (4,) if inlinable else (8, 8)

        class MM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(shape))

            def forward(self, x):
                return x @ self.param

        dtype = torch.float16

        # Populate a cache entry.
        mod1 = MM().to(device=device, dtype=dtype)
        with torch.no_grad():
            x = torch.rand(shape).to(device=device, dtype=dtype)
            out0 = mod1(x)
            out1 = torch.compile(mod1)(x)
            self.assertEqual(out0, out1)

        self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        counters.clear()
        self.reset()

        # Same nn.Module, but with different parameters. In the case that the param can
        # be inlined, we should consider the actual tensor value and we expect a cache
        # miss (because the values are different here). If the param cannot be inlined,
        # then we consider only the tensor metadata and we expect a cache hit.
        mod2 = MM().to(device=device, dtype=dtype)
        self.assertNotEqual(mod1.param, mod2.param)

        with torch.no_grad():
            x = torch.rand(shape).to(device=device, dtype=dtype)
            out0 = mod2(x)
            out1 = torch.compile(mod2)(x)
            self.assertEqual(out0, out1)

        self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
        self.assertEqual(
            counters["inductor"]["fxgraph_cache_miss"], 1 if inlinable else 0
        )
        self.assertEqual(
            counters["inductor"]["fxgraph_cache_hit"], 0 if inlinable else 1
        )


@instantiate_parametrized_tests
class TestStandaloneCompile(TestCase):
    def setUp(self):
        super().setUp()
        counters.clear()
        PatchCaches.setUp()
        CacheArtifactManager.clear()

    def tearDown(self):
        super().tearDown()
        PatchCaches.tearDown()

    def reset(self):
        AOTAutogradCache.clear()
        PyCodeCache.cache_clear(purge=True)
        torch._dynamo.reset()
        clear_caches()

    def capture(self, fn, dynamic=None):
        def inner(*args):
            gm = None
            actual_args = None
            kwargs = None

            def backend(gm_, args_, **kwargs_):
                nonlocal gm
                nonlocal actual_args
                nonlocal kwargs
                gm = gm_
                actual_args = args_
                kwargs = kwargs_
                return gm

            _ = torch.compile(fn, fullgraph=True, backend=backend, dynamic=dynamic)(
                *args
            )
            return gm, actual_args, kwargs

        return inner

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("format", ("binary", "unpacked"))
    @parametrize("dynamic", (False, True))
    @parametrize("graph_partition", (False, True))
    @parametrize("is_aot", (False, True))
    def test_basic(
        self,
        device: str,
        format: str,
        dynamic: bool,
        graph_partition: bool,
        is_aot: bool,
    ) -> None:
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        # AOT mode does not support unpacked format
        if is_aot and format == "unpacked":
            raise unittest.SkipTest("AOT mode does not support unpacked format")

        mod = torch.nn.Linear(1, 3, device=device)
        x = torch.randn(4, 1, device=device)
        if dynamic:
            torch._dynamo.mark_dynamic(x, 0)

        def f(x):
            with torch.no_grad():
                return mod(x), x.sin()

        eager_out = f(x)

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            config.patch(graph_partition=graph_partition),
        ):
            path = (
                temp_dir
                if format == "unpacked"
                else os.path.join(temp_dir, "compiled_artifact.bin")
            )
            with fresh_cache():
                gm, args, kwargs = self.capture(f)(x)
                assert not kwargs

                compiled_artifact = torch._inductor.standalone_compile(
                    gm, args, aot=is_aot
                )
                compiled_artifact.save(path=path, format=format)

            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            with fresh_cache():
                loaded = torch._inductor.CompiledArtifact.load(path=path, format=format)
                if dynamic:
                    concrete_args = [
                        4 if isinstance(a, torch.SymInt) else a for a in args
                    ]
                else:
                    concrete_args = args
                compiled_out = loaded(*concrete_args)
                self.assertEqual(eager_out, compiled_out)

            if not is_aot:
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("dynamic", (False, True))
    @parametrize("is_aot", (False, True))
    def test_call_in_backend(self, dynamic: bool, is_aot: bool) -> None:
        mod = torch.nn.Linear(1, 3)
        x = torch.randn(4, 1)
        if dynamic:
            torch._dynamo.mark_dynamic(x, 0)

        def f(x):
            with torch.no_grad():
                return mod(x)

        eager_out = f(x)

        def backend(gm, args, **kwargs):
            return torch._inductor.standalone_compile(gm, args, aot=is_aot)

        with fresh_cache():
            compiled_out = torch.compile(f, fullgraph=True, backend=backend)(x)
            self.assertEqual(eager_out, compiled_out)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_save_in_new_path(self) -> None:
        mod = torch.nn.Linear(1, 3)
        x = torch.randn(4, 1)

        def f(x):
            with torch.no_grad():
                return mod(x)

        eager_out = f(x)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "new_dir")
            with fresh_cache():
                gm, args, kwargs = self.capture(f)(x)
                assert not kwargs

                compiled_artifact = torch._inductor.standalone_compile(gm, args)
                compiled_artifact.save(path=path, format="unpacked")

            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            with fresh_cache():
                loaded = torch._inductor.CompiledArtifact.load(
                    path=path, format="unpacked"
                )
                compiled_out = loaded(*args)[0]
                self.assertEqual(eager_out, compiled_out)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("device", (GPU_TYPE, "cpu"))
    def test_modify_unpacked_file(self, device: str) -> None:
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        x = torch.ones(4, device=device)

        def f(x):
            with torch.no_grad():
                return 2 * x, x.sin()

        eager_out = f(x)

        with tempfile.TemporaryDirectory() as temp_dir:
            with fresh_cache():
                gm, args, kwargs = self.capture(f)(x)
                assert not kwargs

                compiled_artifact = torch._inductor.standalone_compile(gm, args)
                compiled_out = compiled_artifact(*args)
                self.assertEqual(eager_out, compiled_out)

                compiled_artifact.save(path=temp_dir, format="unpacked")

            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            with fresh_cache():
                # Now modify the output file and expect to see the changes
                for subdir in os.listdir(temp_dir):
                    if subdir in ["aotautograd", "fxgraph"]:
                        continue
                    subdir_path = os.path.join(temp_dir, subdir)
                    for file in os.listdir(subdir_path):
                        file_path = os.path.join(subdir_path, file)
                        assert os.path.isfile(file_path)
                        with open(file_path) as f:
                            file_contents = f.read()
                        if device == GPU_TYPE:
                            file_contents = file_contents.replace(
                                "tmp1 = 2.0", "tmp1 = 8.0"
                            )
                        else:
                            assert device == "cpu"
                            file_contents = file_contents.replace(
                                "auto tmp1 = static_cast<float>(2.0);",
                                "auto tmp1 = static_cast<float>(8.0);",
                            )
                        with open(file_path, "w") as f:
                            f.write(file_contents)

                loaded = torch._inductor.CompiledArtifact.load(
                    path=temp_dir, format="unpacked"
                )
                compiled_out = loaded(*args)
                self.assertEqual(4 * eager_out[0], compiled_out[0])

            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @unittest.skipIf(IS_FBCODE, "torch import error")
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_different_process(self):
        x = torch.ones(4, 1)

        def f(x):
            return x.sin() * 2

        gm, args, kwargs = self.capture(f)(x)
        assert not kwargs

        with tempfile.TemporaryDirectory() as temp_dir:
            path = normalize_path_separator(
                os.path.join(temp_dir, "compiled_artifact.bin")
            )

            with fresh_cache():
                compiled_artifact = torch._inductor.standalone_compile(gm, args)
                compiled_artifact.save(path=path)

            script = f"""
import torch
from torch._inductor.utils import fresh_cache

arg = torch.ones(4, 1)
with fresh_cache():
    loaded = torch._inductor.CompiledArtifact.load(path="{path}")
    compiled_result = loaded(arg)[0]

eager_result = arg.sin() * 2

if not torch.allclose(eager_result, compiled_result, atol=0.1, rtol=0.01):
    raise RuntimeError("tensors do not match")
"""
            try:
                subprocess.check_output(
                    [sys.executable, "-c", script],
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.realpath(__file__)),
                )
            except subprocess.CalledProcessError as e:
                self.fail(
                    msg=(
                        "Subprocess exception while attempting to run test: "
                        + e.output.decode("utf-8")
                    )
                )

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("is_aot", (False, True))
    def test_dynamic_shapes_from_graph(self, is_aot: bool):
        def f(x):
            return x.shape[0] * x

        x = torch.ones(3)
        torch._dynamo.mark_dynamic(x, 0)
        with fresh_cache():
            # captured graph is lambda s0, x: x * s0
            gm, args, kwargs = self.capture(f)(x)
            assert not kwargs

        compiled_artifact = torch._inductor.standalone_compile(
            gm, args, dynamic_shapes="from_graph", aot=is_aot
        )
        x = torch.ones(4)
        (result,) = compiled_artifact(4, x)
        self.assertEqual(result, x * 4)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_normalize_inputs": True})
    @parametrize("is_aot", (False, True))
    def test_split_module(self, is_aot):
        class Mod(torch.nn.Module):
            def forward(self, x, a0, a1, b0, b1, c0, c1):
                x = x + (a0**2) + (a1 / 2)
                x = x + (b0**2) + (b1 / 2)
                x = x + (c0**2) + (c1 / 2)
                return x

        seen = 0
        splits = [4, 8]

        def split(n):
            nonlocal seen
            if seen < splits[0]:
                seen += 1
                return 0
            elif seen < splits[1]:
                seen += 1
                return 1
            else:
                seen += 1
                return 2

        def t():
            return torch.randn([])

        x = t()
        a0 = t()
        a1 = t()
        b0 = t()
        b1 = t()
        c0 = t()
        c1 = t()

        example_inputs = (x, a0, a1, b0, b1, c0, c1)
        gm, inps, _ = self.capture(Mod())(*example_inputs)
        split = torch.fx.passes.split_module.split_module(gm, gm, split)

        # Each of the split graphs only has one output.
        ca0 = torch._inductor.standalone_compile(
            split.submod_0, (a0, x, a1), aot=is_aot
        )
        ca1 = torch._inductor.standalone_compile(
            split.submod_1, (b0, x, b1), aot=is_aot
        )
        ca2 = torch._inductor.standalone_compile(
            split.submod_2, (c0, x, c1), aot=is_aot
        )

        y = ca0(a0, x, a1)
        y = ca1(b0, y, b1)
        y = ca2(c0, y, c1)
        if not is_aot:
            # fx graph cache doesn't run in AOT mode
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)
        # TODO: split_module causes ca1 and ca2 to have different type annotations
        # for the parameter x, so we can only AOTAutogradCache cache hit once instead of twice
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        expected = Mod()(*example_inputs)
        self.assertEqual(y, expected)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("is_aot", (False, True))
    @parametrize("config_patches", [True, False])
    def test_dynamic_shapes_from_example_inputs(self, config_patches, is_aot):
        def f(x):
            return x.shape[0] * x

        x = torch.ones(3)
        torch._dynamo.mark_dynamic(x, 0)
        with fresh_cache():
            # captured graph is lambda s0, x: x * s0
            gm, args, kwargs = self.capture(f)(x)
            assert not kwargs

        if config_patches:
            config_patches = {"fx_graph_cache": True}
        else:
            config_patches = None

        # specialized on example inputs
        compiled_artifact = torch._inductor.standalone_compile(
            gm,
            (5, torch.ones(4)),
            dynamic_shapes="from_example_inputs",
            options={"config_patches": config_patches},
            aot=is_aot,
        )
        x = torch.ones(4)
        (result,) = compiled_artifact(3, x)
        # int 5 was baked in!
        self.assertEqual(result, x * 5)

        # size 4 was baked in
        with self.assertRaisesRegex(AssertionError, "expected size 5==4"):
            x = torch.randn(5)
            (result,) = compiled_artifact(4, x)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("is_aot", (True, False))
    @parametrize("dynamic_shapes", ["from_graph", "from_example_inputs"])
    def test_static_shapes(self, dynamic_shapes, is_aot):
        def f(x):
            return x.shape[0] * x

        static_x = torch.randn(3)
        with fresh_cache():
            # static_gm is lambda x: x * 3
            static_gm, args, kwargs = self.capture(f, dynamic=False)(static_x)
            assert not kwargs
        compiled_artifact = torch._inductor.standalone_compile(
            static_gm, [static_x], dynamic_shapes=dynamic_shapes, aot=is_aot
        )
        x = torch.randn(3)
        (result,) = compiled_artifact(x)
        self.assertEqual(result, x * 3)
        with self.assertRaisesRegex(AssertionError, "expected size 4==3"):
            x = torch.randn(4)
            (result,) = compiled_artifact(x)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("is_aot", (True, False))
    @parametrize("dynamic_shapes", ["from_tracing_context", "from_graph"])
    def test_backend(self, dynamic_shapes, is_aot):
        def f(x):
            return x.shape[0] * x

        x = torch.randn(3)
        torch._dynamo.mark_dynamic(x, 0)

        def backend(gm, args, **kwargs):
            compiled_artifact = torch._inductor.standalone_compile(
                gm, args, dynamic_shapes=dynamic_shapes, aot=is_aot
            )
            y = torch.randn(4)
            (result,) = compiled_artifact(4, y)
            self.assertEqual(result, y * 4)
            return compiled_artifact

        torch._dynamo.reset()
        _ = torch.compile(f, backend=backend)(x)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("is_aot", (True, False))
    def test_backend_dynamic_shapes_from_example_inputs(self, is_aot):
        def f(x):
            return x.shape[0] * x

        x = torch.ones(4)
        torch._dynamo.mark_dynamic(x, 0)

        def backend(gm, args, **kwargs):
            compiled_artifact = torch._inductor.standalone_compile(
                gm, [5, torch.ones(4)], dynamic_shapes="from_example_inputs", aot=is_aot
            )
            y = torch.ones(4)
            (result,) = compiled_artifact(4, y)
            # 5 was baked in
            self.assertEqual(result, y * 5)

            # shape of y was baked in
            with self.assertRaisesRegex(AssertionError, "expected size 5==4"):
                y = torch.ones(5)
                (result,) = compiled_artifact(4, y)

            return compiled_artifact

        torch._dynamo.reset()
        _ = torch.compile(f, backend=backend)(x)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize(
        "dynamic_shapes", ["from_tracing_context", "from_graph", "from_example_inputs"]
    )
    def test_backend_static_shapes(self, dynamic_shapes):
        # on static_x, all of these options should produce a static graph,
        # but it's a bit hard to tell, so these are just smoke tests.
        static_x = torch.randn(3)

        def f(x):
            return x.shape[0] * x

        def backend(gm, args, **kwargs):
            return torch._inductor.standalone_compile(
                gm, args, dynamic_shapes=dynamic_shapes
            )

        result = torch.compile(f, backend=backend)(static_x)
        self.assertEqual(result, static_x * 3)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_custom_pass_handling(self):
        """
        Test that properly-registered custom hooks allow caching.
        """

        class TestCustomGraphPass(CustomGraphPass):
            def __call__(self, graph: torch.fx.graph.Graph) -> None:
                return None

            def uuid(self) -> Optional[Union[bytes, str]]:
                return "uuid"

        def fn(a, b):
            return torch.mm(a, b)

        a = torch.rand(8, 32, device="cpu")
        b = torch.rand(32, 8, device="cpu")
        compiled_fn = torch.compile(fn)

        # The cache should be bypassed if a custom hook doesn't use CustomGraphPass.
        with config.patch({"post_grad_custom_pre_pass": lambda x: x}):
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # With proper usage, we expect normal caching.
        custom_pass = TestCustomGraphPass()
        with config.patch(
            {
                "post_grad_custom_pre_pass": custom_pass,
                "post_grad_custom_post_pass": custom_pass,
                "joint_custom_pre_pass": custom_pass,
                "joint_custom_post_pass": custom_pass,
            }
        ):
            self.reset()
            counters.clear()

            # Cache miss
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            self.reset()
            counters.clear()

            # Cache hit
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)


class TestCustomPartitionerFn(CustomPartitionerFn):
    def __init__(self):
        self._uuid = None

    def __call__(
        self, gm, joint_inputs, **kwargs
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        return gm, gm  # Dummy implementation

    def uuid(self) -> Optional[Union[bytes, str]]:
        return self._uuid


class TestFxGraphCacheHashing(TestCase):
    def test_parameter_constants(self):
        """
        Test the hashing of parameter constants.
        """
        small = torch.nn.Parameter(torch.rand(8))
        large = torch.nn.Parameter(torch.rand(32))

        self.assertTrue(GraphLowering.can_inline_constant(small))
        self.assertFalse(GraphLowering.can_inline_constant(large))

        # By default, we hash the metadata and values independent of the size.
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        data = pickler.dumps(small)
        self.assertIsInstance(pickle.loads(data), TensorMetadataAndValues)
        data = pickler.dumps(large)
        self.assertIsInstance(pickle.loads(data), TensorMetadataAndValues)

        # For frozen parameters, we only hash the values of small tensors.
        gm._has_frozen_params = True
        gm._frozen_param0 = small
        gm._frozen_param1 = large
        small._is_frozen_param = True
        large._is_frozen_param = True
        pickler = FxGraphCachePickler(gm)

        data = pickler.dumps(small)
        self.assertIsInstance(pickle.loads(data), TensorMetadataAndValues)
        data = pickler.dumps(large)
        self.assertIsInstance(pickle.loads(data), TensorMetadata)

    def test_hash_fake_tensors(self):
        """
        Test hashing (pickling) FakeTensors with various characteristics.
        """
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)
        with torch._subclasses.FakeTensorMode():
            # Verify that FakeTensors get pickled into a TensorMetadata:
            data = pickler.dumps(torch.randn(1))
            self.assertIsInstance(pickle.loads(data), TensorMetadata)

            # Different shapes:
            self.assertEqual(
                pickler.dumps(torch.randn(3)),
                pickler.dumps(torch.randn(3)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3)),
                pickler.dumps(torch.randn(4)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3)),
                pickler.dumps(torch.randn(3, 3)),
            )

            self.assertEqual(
                pickler.dumps(torch.randn(3, 3)),
                pickler.dumps(torch.randn(3, 3)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3, 3)),
                pickler.dumps(torch.randn(3, 4)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3, 3)),
                pickler.dumps(torch.randn(4, 3)),
            )

            # Different strides:
            self.assertEqual(
                pickler.dumps(torch.randn(3, 3)),
                pickler.dumps(torch.randn(3, 3).transpose(0, 1).transpose(0, 1)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3, 3)),
                pickler.dumps(torch.randn(3, 3).transpose(0, 1)),
            )

            # Different storage offsets:
            self.assertEqual(
                pickler.dumps(torch.randn(3)[1:]),
                pickler.dumps(torch.randn(3)[1:]),
            )
            self.assertEqual(
                pickler.dumps(torch.randn(3)[1:]),
                pickler.dumps(torch.randn(2)),
            )

            # Different dtypes:
            self.assertEqual(
                pickler.dumps(torch.randn(3, dtype=torch.float32)),
                pickler.dumps(torch.randn(3, dtype=torch.float32)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3, dtype=torch.float32)),
                pickler.dumps(torch.randn(3, dtype=torch.float64)),
            )

            # Different 'requires_grad':
            self.assertEqual(
                pickler.dumps(torch.randn(3, requires_grad=True)),
                pickler.dumps(torch.randn(3, requires_grad=True)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3, requires_grad=True)),
                pickler.dumps(torch.randn(3, requires_grad=False)),
            )

            # Different memory formats:
            self.assertNotEqual(
                pickler.dumps(torch.randn(1, 2, 3, 4)),
                pickler.dumps(
                    torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last)
                ),
            )

            # Different devices:
            self.assertEqual(
                pickler.dumps(torch.randn(3, device="meta")),
                pickler.dumps(torch.randn(3, device="meta")),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3, device="meta")),
                pickler.dumps(torch.randn(3, device="cpu")),
            )

            if HAS_MULTIGPU:
                self.assertEqual(
                    pickler.dumps(torch.randn(3, device=f"{GPU_TYPE}:1")),
                    pickler.dumps(torch.randn(3, device=f"{GPU_TYPE}:1")),
                )
                self.assertNotEqual(
                    pickler.dumps(torch.randn(3, device=f"{GPU_TYPE}:0")),
                    pickler.dumps(torch.randn(3, device=f"{GPU_TYPE}:1")),
                )

    def test_hash_kwargs(self):
        """
        Test the special handling of the kwargs when hashing, i.e.,
        ordering of the kwargs dict and any set arguments.
        """
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        # Dict order of the kwargs should not affect hashes.
        details1 = FxGraphHashDetails(None, [], {"a": 0, "z": 1}, [])
        details2 = FxGraphHashDetails(None, [], {"z": 1, "a": 0}, [])
        self.assertEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )

        # Different kwarg values should affect hashes.
        details1 = FxGraphHashDetails(None, [], {"a": 0}, [])
        details2 = FxGraphHashDetails(None, [], {"a": 1}, [])
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )

        # Set order should not affect hashes. Sets are unordered, but
        # sorting and creating a new set seems to change the order.
        set1 = {"a", "b", "c", "d", "e", "f", "g"}
        set2 = set(sorted(set1))  # noqa: C414
        details1 = FxGraphHashDetails(None, [], {"a": set1}, [])
        details2 = FxGraphHashDetails(None, [], {"a": set2}, [])
        self.assertEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )

        # But different set contents should affect hashes.
        details1 = FxGraphHashDetails(None, [], {"a": {1, 2, 3}}, [])
        details2 = FxGraphHashDetails(None, [], {"a": {1, 2}}, [])
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
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

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        self.assertEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details3),
        )

    def test_hash_private_config_changes(self):
        """
        Test that private config settings affect hashes.
        """
        with config.patch({"_micro_pipeline_tp": False}):
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

        with config.patch({"_micro_pipeline_tp": True}):
            details3 = FxGraphHashDetails(None, [], {}, [])

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        self.assertEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details3),
        )

    def test_non_serializable_custom_passes_causes_cache_miss(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 4))

            def forward(self, x):
                return x @ self.param

        mod1 = Mod()
        mod_compiled = torch.compile(mod1)
        with torch.no_grad():
            x = torch.rand(4, 4)
            # miss
            mod_compiled(x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            # hit
            torch._dynamo.reset()
            mod_compiled(x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            torch._dynamo.reset()
            counters.clear()

            # hit
            mod_compiled(x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            with config.patch({"_fuse_ddp_communication_passes": ["new_pass_foo_bar"]}):
                # miss (private config changed)
                torch._dynamo.reset()
                mod_compiled(x)
                self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
                torch._dynamo.reset()
                counters.clear()

            with (
                capture_logs("torch._inductor.codecache", logging.INFO) as logs,
                config.patch({"_fuse_ddp_communication_passes": [lambda *args: None]}),
            ):
                # bypass (custom pass is not serializable)
                mod_compiled(x)
                self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 1)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                counters.clear()
            # assert that our bypass is explicit
            self.assertTrue(
                any(
                    x.getMessage()
                    == "Bypassing FX Graph Cache because 'Unsupported _fuse_ddp_communication_pass'"
                    for x in logs
                )
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

            gm = torch.fx.GraphModule({}, torch.fx.Graph())
            pickler = FxGraphCachePickler(gm)

            self.assertEqual(
                pickler.dumps(details1),
                pickler.dumps(details2),
            )
            self.assertNotEqual(
                pickler.dumps(details1),
                pickler.dumps(details3),
            )

    def test_hash_custom_backend_pass(self):
        """
        Test CustomGraphModulePass usage.
        """

        class TestCustomGraphModulePass(CustomGraphModulePass):
            def __init__(self):
                self._uuid = None

            def __call__(self, gm: torch.fx.GraphModule) -> None:
                return None

            def uuid(self) -> Optional[Union[bytes, str]]:
                return self._uuid

        custom_pass = TestCustomGraphModulePass()
        with patch_inductor_backend("cpu", custom_pass=custom_pass):
            custom_pass._uuid = "1"
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

            custom_pass._uuid = "2"
            details3 = FxGraphHashDetails(None, [], {}, [])

            gm = torch.fx.GraphModule({}, torch.fx.Graph())
            pickler = FxGraphCachePickler(gm)

            self.assertEqual(
                pickler.dumps(details1),
                pickler.dumps(details2),
            )
            self.assertNotEqual(
                pickler.dumps(details1),
                pickler.dumps(details3),
            )

    def test_hash_custom_backend_config(self):
        """
        Test cache correctness when a custom inductor codegen config
        is installed
        """
        with patch_inductor_backend(
            "cpu", custom_backend_config=custom_inductor_config
        ):
            gm = torch.fx.GraphModule({}, torch.fx.Graph())
            pickler = FxGraphCachePickler(gm)
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])
            self.assertEqual(pickler.dumps(details1), pickler.dumps(details2))

            custom_inductor_config.enable_optimisation = True
            details3 = FxGraphHashDetails(None, [], {}, [])
            self.assertNotEqual(pickler.dumps(details2), pickler.dumps(details3))

            torch._dynamo.reset()
            counters.clear()

            custom_inductor_config.enable_optimisation = False
            x = torch.zeros(32)
            y = torch.zeros(32)
            compiled_fn = torch.compile(torch.add)

            compiled_fn(x, y)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            torch._dynamo.reset()
            counters.clear()

            compiled_fn(x, y)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            torch._dynamo.reset()
            counters.clear()

            # Changing the custom config should trigger a recompilation
            custom_inductor_config.enable_optimisation = True
            compiled_fn(x, y)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

    def test_hash_custom_partitioner_fn(self):
        """
        Test that the custom partitioner function's UUID is properly used in the FX graph cache hashing.
        """
        custom_partitioner_fn = TestCustomPartitionerFn()
        with config.patch({"custom_partitioner_fn": custom_partitioner_fn}):
            custom_partitioner_fn._uuid = "1"
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

            custom_partitioner_fn._uuid = "2"
            details3 = FxGraphHashDetails(None, [], {}, [])

            self.assertEqual(details1._custom_partitioner_fn, "1")
            self.assertEqual(details2._custom_partitioner_fn, "1")
            self.assertEqual(details3._custom_partitioner_fn, "2")

            gm = torch.fx.GraphModule({}, torch.fx.Graph())
            pickler = FxGraphCachePickler(gm)

            self.assertEqual(
                pickler.dumps(details1),
                pickler.dumps(details2),
            )
            self.assertNotEqual(
                pickler.dumps(details1),
                pickler.dumps(details3),
            )

    def test_bypass_unsupported(self):
        """
        Test _reduce_unsupported
        """
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        with self.assertRaises(BypassFxGraphCache):
            FxGraphCachePickler(gm).dumps(
                torch.fx.experimental._backward_state.BackwardState()
            )

    def test_stable_strings(self):
        """
        Test that objects containing identical strings pickle the same
        even if they are not the same id.
        """
        s1 = "string"
        s2 = "strin"  # codespell:ignore
        s2 += "g"

        self.assertNotEqual(id(s1), id(s2))

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)
        self.assertEqual(
            pickler.dumps([s1, s1]),
            pickler.dumps([s1, s2]),
        )

    def test_get_hash_for_files(self):
        """
        Test the get_hash_for_files helper.
        """
        # delete=True does not work on Windows.
        # See https://docs.python.org/3.12/library/tempfile.html#tempfile.NamedTemporaryFile
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            try:
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
            finally:
                temp.close()
                os.unlink(temp.name)


class TestCudaCompileCommand(TestCase):
    @requires_cuda_and_triton
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
            cmd_parts: list[str] = check_output_mock.call_args[0][0]
            assert cmd_parts[0].endswith("nvcc"), cmd_parts
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
        PyCodeCache.cache_clear(purge=True)
        torch._dynamo.reset()
        clear_caches()

    @requires_cuda_and_triton
    @unittest.skipIf(not SM80OrLater, "Requires SM80+")
    @unittest.skipIf(
        TEST_WITH_ROCM, "Requires static cuda launcher, which does not support ROCM"
    )
    @config.patch({"use_static_cuda_launcher": True})
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"autotune_local_cache": False})
    @config.patch({"autotune_remote_cache": True})
    @config.patch({"bundled_autotune_remote_cache": False})
    @config.patch({"max_autotune": True})
    @config.patch(
        {"compile_threads": 1}
    )  # Worker processes do not register PatchCaches() properly
    def test_autotune_cache_warm_start(self):
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
            a1 = f_compiled(x, y, a, b)

            self.assertEqual(global_stats.autotune_remote, Stats(2, 0, 2))
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            # Don't reset FxGraphCache, see that it loads again
            torch._dynamo.reset()
            a2 = f_compiled(x, y, a, b)
            self.assertEqual(a1, a2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        self.assertEqual(global_stats.autotune_remote, Stats(2, 2, 2))

        # Check that the cache entries seem reasonable
        for k in global_stats.autotune_remote.cache.keys():
            self.assertRegex(k, r"[0-9a-z]{52}")
        for k in global_stats.triton.cache.keys():
            self.assertRegex(k, r"triton:[0-9a-f]{64}::[0-9a-f]{64}:c[0-9]+")

    @requires_gpu_and_triton
    @unittest.skipIf(not HAS_XPU_AND_TRITON and not SM80OrLater, "Requires SM80+")
    @config.patch({"fx_graph_cache": False})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"autotune_local_cache": False})
    @config.patch({"autotune_remote_cache": True})
    @config.patch({"bundled_autotune_remote_cache": False})
    @config.patch({"max_autotune": True})
    @config.patch(
        {"compile_threads": 1}
    )  # Worker processes do not register PatchCaches() properly
    def test_autotune_cache(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, a, b):
                return x + y, a + b

        def f(x, y, a, b):
            return Model()(x, y, a, b)

        x = torch.randn(100, 100).to(GPU_TYPE)
        y = torch.randn(100, 100).to(GPU_TYPE)
        a = torch.randn(1000, 100).to(GPU_TYPE)
        b = torch.randn(1000, 100).to(GPU_TYPE)
        f_compiled = torch.compile(f, fullgraph=True)

        with PatchCaches():
            f_compiled(x, y, a, b)

            self.assertEqual(global_stats.autotune_remote, Stats(2, 0, 2))

            self.reset()
            f_compiled(x, y, a, b)

        self.assertEqual(global_stats.autotune_remote, Stats(2, 2, 2))

        # Check that the cache entries seem reasonable
        for k in global_stats.autotune_remote.cache.keys():
            self.assertRegex(k, r"[0-9a-z]{52}")
        for k in global_stats.triton.cache.keys():
            self.assertRegex(k, r"triton:[0-9a-f]{64}::[0-9a-f]{64}:c[0-9]+")

    @requires_gpu_and_triton
    @unittest.skipIf(not HAS_XPU_AND_TRITON and not SM80OrLater, "Requires SM80+")
    @config.patch({"fx_graph_cache": False})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"autotune_local_cache": True})
    @config.patch({"autotune_remote_cache": False})
    @config.patch({"bundled_autotune_remote_cache": True})
    @config.patch({"compile_threads": 1})
    @config.patch({"max_autotune": True})
    def test_bundled_autotune_remote_cache(self):
        class Model(torch.nn.Module):
            def forward(self, a, b, c, d, e, f):
                return a + b, c + d, e + f

        def f(a, b, c, d, e, f):
            return Model()(a, b, c, d, e, f)

        f_compiled = torch.compile(f, fullgraph=True)

        a = torch.randn(101, 100).to(GPU_TYPE)
        b = torch.randn(101, 100).to(GPU_TYPE)
        c = torch.randn(102, 100).to(GPU_TYPE)
        d = torch.randn(102, 100).to(GPU_TYPE)
        e = torch.randn(103, 100).to(GPU_TYPE)
        f = torch.randn(103, 100).to(GPU_TYPE)

        with PatchCaches():
            f_compiled(a, b, c, d, e, f)

            self.assertEqual(global_stats.autotune_local, Stats(3, 0, 3))
            self.assertEqual(global_stats.bundled_autotune, Stats(1, 0, 1))

            self.reset()
            f_compiled(a, b, c, d, e, f)

            self.assertEqual(global_stats.autotune_local, Stats(6, 3, 3))
            self.assertEqual(global_stats.bundled_autotune, Stats(1, 1, 1))

            with torch.compiler.config.patch({"cache_key_tag": "test"}):
                global_stats.reset()
                self.reset()
                f_compiled(a, b, c, d, e, f)

                self.assertEqual(global_stats.autotune_local, Stats(3, 0, 3))
                self.assertEqual(global_stats.bundled_autotune, Stats(1, 0, 1))

                self.reset()
                f_compiled(a, b, c, d, e, f)

                self.assertEqual(global_stats.autotune_local, Stats(6, 3, 3))
                self.assertEqual(global_stats.bundled_autotune, Stats(1, 1, 1))

        # Check that the cache entries seem reasonable
        for k in global_stats.autotune_local.cache.keys():
            self.assertRegex(k, r"tmp[^/]*/([^/]{2})/[^/]{64}\.best_config")
        for k in global_stats.bundled_autotune.cache.keys():
            self.assertRegex(k, r"pt2:bundled-autotune-v1::[0-9a-z]{64}:c[0-9]+")
        for k in global_stats.triton.cache.keys():
            self.assertRegex(k, r"triton:[0-9a-f]{64}::[0-9a-f]{64}:c[0-9]+")

    @requires_triton()
    @requires_gpu_and_triton
    @unittest.skipIf(not HAS_XPU_AND_TRITON and not SM80OrLater, "Requires SM80+")
    @config.patch({"fx_graph_cache": False})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"bundled_autotune_remote_cache": False})
    @config.patch({"max_autotune": True})
    @config.patch(
        {"compile_threads": 1}
    )  # Worker processes do not register PatchCaches() properly
    @parametrize("remote_cache", (True, False))
    def test_modified_autotune_cache(self, remote_cache):
        """
        If a developer changes the way the autotune cache is handled,
        there's a chance it'll break the cache. This happened with
        #150122. This test ensures that if torch code changes, then
        old cache entries will be invalidated.
        """

        def mock_torch_key(value: str) -> bytes:
            return value.encode("utf-8")

        def get_autotune_stats():
            if remote_cache:
                return global_stats.autotune_remote
            return global_stats.autotune_local

        def fn(x, y):
            return (x + y).relu()

        x = torch.randn(100, 100).to(GPU_TYPE)
        y = torch.randn(100, 100).to(GPU_TYPE)

        with config.patch(
            {
                "autotune_local_cache": not remote_cache,
                "autotune_remote_cache": remote_cache,
            }
        ):
            with PatchCaches():
                with mock.patch(
                    "torch._inductor.codecache.torch_key",
                    functools.partial(mock_torch_key, "torchkey1"),
                ):
                    f_compiled = torch.compile(fn, fullgraph=True)
                    res1 = f_compiled(x, y)

                self.assertEqual(get_autotune_stats(), Stats(1, 0, 1))

                torch._dynamo.reset()
                PyCodeCache.cache_clear()

                with mock.patch(
                    "torch._inductor.codecache.torch_key",
                    functools.partial(mock_torch_key, "torchkey2"),
                ):
                    f_compiled = torch.compile(fn, fullgraph=True)
                    res2 = f_compiled(x, y)

                self.assertEqual(get_autotune_stats(), Stats(2, 0, 2))

                self.assertEqual(res1, res2)


class TestRemoteAOTAutogradCache(TestCase):
    @requires_gpu()
    @unittest.skipIf(not HAS_XPU_AND_TRITON and not SM80OrLater, "Requires SM80+")
    @config.patch({"fx_graph_cache": False})
    @config.patch({"fx_graph_remote_cache": True})
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @torch._functorch.config.patch({"enable_remote_autograd_cache": True})
    def test_autograd_remote_cache(self):
        def f(a, b):
            return a + b

        f_compiled = torch.compile(f)
        a = torch.randn(101, 100, device=GPU_TYPE, requires_grad=False)
        b = torch.randn(101, 100, device=GPU_TYPE, requires_grad=False)
        with PatchCaches():
            f_compiled(a, b)

            self.assertEqual(global_stats.aot_autograd, Stats(1, 0, 1))
            self.assertEqual(global_stats.fx_graph, Stats(1, 0, 1))

            torch._dynamo.reset()

            f_compiled(a, b)
            self.assertEqual(global_stats.aot_autograd, Stats(1, 1, 1))
            self.assertEqual(global_stats.fx_graph, Stats(1, 1, 1))

            torch._dynamo.reset()

            with torch.compiler.config.patch({"cache_key_tag": "test"}):
                f_compiled(a, b)
            self.assertEqual(global_stats.aot_autograd, Stats(2, 1, 2))
            self.assertEqual(global_stats.fx_graph, Stats(2, 1, 2))

        # Check that the cache entries seem reasonable
        for k in global_stats.aot_autograd.cache.keys():
            self.assertRegex(k, r"pt2:autograd-experimental::[0-9a-z]{52}:c[0-9]+")

        for k in global_stats.fx_graph.cache.keys():
            self.assertRegex(k, r"pt2:fx-graph-v1::[0-9a-z]{52}:c[0-9]+")

    @requires_gpu_and_triton
    @unittest.skipIf(not HAS_XPU_AND_TRITON and not SM80OrLater, "Requires SM80+")
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
    def test_fresh_cache(self):
        def fn(x, y):
            return x + y

        a = torch.rand(10)
        b = torch.rand(10)

        with fresh_cache():
            self.assertEqual(len(PyCodeCache.modules), 0)
            res1 = torch.compile(fn)(a, b)
            cache_dir1 = cache_dir()

        torch._dynamo.reset()
        with fresh_cache():
            self.assertEqual(len(PyCodeCache.modules), 0)
            res2 = torch.compile(fn)(a, b)
            cache_dir2 = cache_dir()

        self.assertEqual(res1, res2)
        self.assertNotEqual(cache_dir1, cache_dir2)

    # This combination of settings exposed a bug where we cleared the
    # PyCodeCache disk artifacts while they were still needed:
    @requires_gpu_and_triton
    @config.patch(
        {
            "coordinate_descent_tuning": True,
            "force_disable_caches": True,
        }
    )
    def test_force_disable_coordinate_descent(self):
        def fn():
            inp = torch.randn(32, 50, 768, device=GPU_TYPE)
            weight = torch.randn(768, 768, device=GPU_TYPE)
            layer = torch.nn.LayerNorm(768, device=GPU_TYPE)
            return layer(inp @ weight)

        torch.compile(fn)()


if __name__ == "__main__":
    run_tests()
