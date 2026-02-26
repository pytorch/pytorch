# Owner(s): ["module: dynamo"]

import copy
import dataclasses
import functools
import os
import pickle
import shutil
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._functorch._aot_autograd
from torch._dynamo import config as dynamo_config
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.autograd_cache import (
    AOTAutogradCache,
    autograd_cache_key,
    BypassAOTAutogradCache,
    sanitize_gm_for_cache,
)
from torch._functorch._aot_autograd.schemas import AOTConfig
from torch._guards import TracingContext
from torch._inductor import config as inductor_config
from torch._inductor.custom_graph_pass import CustomRuntimeEstimator
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.runtime.triton_compat import tl, triton
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import fresh_cache
from torch._subclasses import FakeTensorMode
from torch.compiler._cache import CacheArtifactManager
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_cuda import SM80OrLater, TEST_MULTIGPU
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfWindows,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_triton
from torch.testing._internal.triton_utils import requires_cuda_and_triton
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


def aot_eager_regional_inductor():
    """
    Regional inductor backend for AOT autograd.
    Uses regional_inductor as both forward and backward compiler.
    """
    from torch._dynamo.backends.common import aot_autograd
    from torch.fx.passes.regional_inductor import regional_inductor

    return aot_autograd(
        fw_compiler=regional_inductor,
        bw_compiler=regional_inductor,
    )


def saved_tensors_hooks_to_gm(
    pack_fn,
    unpack_fn,
    pack_cache_hash=None,
    unpack_cache_hash=None,
    symbolic_tracing=True,
    inp_fn=None,
):
    if symbolic_tracing:
        pack_gm = torch.fx.symbolic_trace(pack_fn)
        unpack_gm = torch.fx.symbolic_trace(unpack_fn)
    else:
        from functorch import make_fx

        if inp_fn:
            inp = inp_fn()
        else:
            inp = torch.randn(2, 3)
            torch._dynamo.mark_dynamic(inp, 0)
            torch._dynamo.mark_dynamic(inp, 1)
        pack_out = pack_fn(inp)
        pack_gm = make_fx(pack_fn)(inp)
        unpack_gm = make_fx(unpack_fn)(pack_out)

    def set_manual_hash(g, manual_hash):
        for node in g.nodes:
            if node.meta and node.meta.get("is_wrapped", False):
                node.meta["user_cache_hash"] = manual_hash

    if pack_cache_hash:
        set_manual_hash(pack_gm.graph, pack_cache_hash)
    if unpack_cache_hash:
        set_manual_hash(unpack_gm.graph, unpack_cache_hash)
    return pack_gm, unpack_gm


def amax_to_scale(
    amax: torch.Tensor,
    float8_dtype: torch.dtype,
    round_scales_to_power_of_2: bool = False,
):
    amax = amax.to(torch.float64)
    res = torch.finfo(float8_dtype).max / torch.clamp(amax, min=1e-12)
    res = res.to(torch.float32)
    return res


# Must be at module level to use fx.wrap
@torch.fx.wrap
def _pack_fp8_with_scale_wrap(x):
    if not x.dtype.is_floating_point:
        return x

    amax = torch.max(torch.abs(x))
    scale = amax_to_scale(amax, torch.float8_e5m2)
    x_scaled = x.to(torch.float32) * scale
    x_fp8 = x_scaled.to(torch.float8_e5m2)
    return x.dtype, scale, x_fp8


@torch.fx.wrap
def _unpack_fp8_with_scale_wrap(x):
    if isinstance(x, torch.Tensor):
        return x

    dtype, scale, x_fp8 = x
    y = x_fp8.to(torch.float32) / scale
    return y.to(dtype)


@instantiate_parametrized_tests
class AOTAutogradCacheTests(InductorTestCase):
    def setUp(self):
        """
        Reset all counters and caches before each unit test
        """
        super().setUp()
        counters.clear()
        self._clear_all_caches()

    def _clear_all_caches(self):
        """
        Clear every cache, including AOTAutogradCache and FXCache
        """
        torch._inductor.codecache.FxGraphCache.clear()
        AOTAutogradCache.clear()
        CacheArtifactManager.clear()
        self._clear_dynamo_and_codecache()

    def _clear_dynamo_and_codecache(self):
        """
        Clear unrelated caches, like dynamo and PyCodeCache
        """
        torch._dynamo.reset()
        torch._inductor.codecache.PyCodeCache.cache_clear(purge=True)

    @requires_triton()
    @functorch_config.patch({"enable_autograd_cache": True})
    @inductor_config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "autotune_local_cache": True,
        }
    )
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
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

        a = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)
        b = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)

        # Record artifacts
        with fresh_cache():
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            # A first call should miss in the cache.
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            if hasattr(a, "_dynamo_weak_dynamic_indices"):
                del a._dynamo_weak_dynamic_indices
            self.assertEqual(eager_result, compiled_result)
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            else:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        autotune_expect = 2 if device == GPU_TYPE else 0

        if functorch_config.bundled_autograd_cache:
            self.assertEqual(len(cache_info.inductor_artifacts), 0)
        else:
            self.assertEqual(len(cache_info.inductor_artifacts), 2)
        self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
        self.assertEqual(len(cache_info.pgo_artifacts), 0)

        self._clear_all_caches()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # We did not load anything so dont hit yet
        with fresh_cache():
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            compiled_result.sum().backward()
            if hasattr(a, "_dynamo_weak_dynamic_indices"):
                del a._dynamo_weak_dynamic_indices
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            else:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 4)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        self._clear_all_caches()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with fresh_cache():
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(len(cache_info.inductor_artifacts), 0)
            else:
                self.assertEqual(len(cache_info.inductor_artifacts), 2)
            self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
            self.assertEqual(len(cache_info.pgo_artifacts), 0)

            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            if hasattr(a, "_dynamo_weak_dynamic_indices"):
                del a._dynamo_weak_dynamic_indices
            self.assertEqual(eager_result, compiled_result)
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            else:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 4)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
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
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a, b))

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_vmap(self):
        """
        make
        """

        def fn(x, y):
            f = lambda x, y: (x * y + 1).sum(dim=0)  # noqa: E731
            vmapped = torch.vmap(f)(x, y)
            return vmapped.sum(dim=0)

        x = torch.randn(25, requires_grad=True)
        y = torch.randn(25, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        y2 = y.detach().clone().requires_grad_(True)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(x, y), compiled_fn(x2, y2))
        fn(x, y).sum().backward()
        compiled_fn(x2, y2).sum().backward()
        self.assertEqual(x.grad, x2.grad)
        self.assertEqual(y.grad, y2.grad)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Reset all tensors
        x = torch.randn(25, requires_grad=True)
        y = torch.randn(25, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        y2 = y.detach().clone().requires_grad_(True)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(x, y), compiled_fn(x2, y2))
        fn(x, y).sum().backward()
        compiled_fn(x2, y2).sum().backward()
        self.assertEqual(x.grad, x2.grad)
        self.assertEqual(y.grad, y2.grad)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_multi_graph_specialization(self):
        """
        Verify multi graph specializations all cache hit
        """

        def fn(x):
            return x * 5

        a = torch.randn(5)
        a8 = torch.randn(8)
        a16 = torch.randn(16)
        torch._dynamo.mark_dynamic(
            a,
            0,
            specialize_on=[
                lambda x: x == 8,
                lambda x: x == 16,
            ],
        )

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        compiled_fn(a)
        compiled_fn(a8)
        compiled_fn(a16)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 3)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 3)

        self._clear_dynamo_and_codecache()

        # A second call should hit on all 3 graphs
        compiled_fn(a)
        compiled_fn(a8)
        compiled_fn(a16)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 3)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 3)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 3)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_symbol_specialization(self):
        """
        Verify the symbol specializations don't cause cache miss.
        """

        def fn(x, y, z):
            return (torch.randn(5) + x + y, z * torch.randn(1))

        a = torch.rand(5)
        torch._dynamo.maybe_mark_dynamic(a, 0)
        b = torch.rand(5)
        c = torch.randn(6)
        torch._dynamo.maybe_mark_dynamic(c, 0)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        compiled_fn(a, b, c)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # A second call should hit even if a new dimension is marked as dynamic
        # that is later specialized as part of tracing.
        a = torch.rand(5)
        torch._dynamo.maybe_mark_dynamic(a, 0)
        b = torch.rand(5)
        torch._dynamo.maybe_mark_dynamic(b, 0)
        c = torch.randn(6)
        torch._dynamo.maybe_mark_dynamic(c, 0)
        self._clear_dynamo_and_codecache()

        compiled_fn(a, b, c)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @functorch_config.patch({"enable_autograd_cache": True})
    def test_aot_runtime_trace_joint(self):
        @torch.compile(backend="inductor")
        def f(x):
            tmp = x.sin()
            s0 = tmp.shape[0]
            return tmp.expand(s0, s0)

        x_a = torch.randn(4, requires_grad=True)
        x = TwoTensor(x_a, x_a.clone())
        out = f(x)
        out.sum().backward()

        self._clear_dynamo_and_codecache()
        out = f(x)
        out.sum().backward()

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @skipIfWindows(
        msg="Known issue: Window can't delete loaded modules, so we can't clear module cache."
    )
    def test_clear_fx_graph_cache(self):
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
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear FX graph cache: second call should also be a miss
        self._clear_dynamo_and_codecache()
        torch._inductor.codecache.FxGraphCache.clear()
        self.assertEqual(fn(a, b), compiled_fn(a, b))

        if functorch_config.bundled_autograd_cache:
            # Bundled AutogradCache doesn't care if FxGraphCache is cleared
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        else:
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            # We save again into the cache
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"strict_autograd_cache": True})
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @requires_triton()
    def test_non_bundled_to_bundled_config_change(self):
        if functorch_config.bundled_autograd_cache:
            raise unittest.SkipTest("BundledAutogradCache is already enabled")

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25, device=GPU_TYPE)
        b = torch.rand(5, 5, device=GPU_TYPE)

        compiled_fn = torch.compile(fn, backend="inductor")
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Now turn on bundled autograd cache, see that we successfully save again
        with functorch_config.patch({"bundled_autograd_cache": True}):
            torch._dynamo.reset()
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "view_replay_for_aliased_outputs": True}
    )
    def test_view_replay(self):
        def fn(a):
            tmp = a.detach()
            a.mul_(2)
            return a, tmp

        with torch.autograd._force_original_view_tracking(True):
            compiled_fn = torch.compile(fn)

        def run_and_check(miss, hit, bypass):
            self._clear_dynamo_and_codecache()

            inp = torch.rand(2, 3)
            compiled_inp = inp.clone().detach()

            with torch.autograd._force_original_view_tracking(True):
                out = fn(inp)
                compiled_out = compiled_fn(compiled_inp)

            self.assertEqual(out, compiled_out)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], miss)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], hit)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], bypass)

        run_and_check(miss=1, hit=0, bypass=0)
        run_and_check(miss=1, hit=1, bypass=0)
        run_and_check(miss=1, hit=2, bypass=0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_invoke_subgraph(self):
        from torch._higher_order_ops.invoke_subgraph import mark_compile_region

        @mark_compile_region
        def gn(x, y):
            return x + y

        @torch.compile(backend="inductor")
        def fn(x, y):
            return gn(x, y) + gn(x, y)

        a = torch.randn(25)
        b = torch.randn(25)

        fn(a, b)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_unsafe_mark_cacheable(self):
        @torch._dynamo.allow_in_graph
        class AllowInGraphFunc(torch.autograd.Function):
            @staticmethod
            def forward(_, x):
                torch._dynamo.graph_break()
                return x.sin()

        @torch.compile
        def fn(x, y, z):
            return AllowInGraphFunc.apply(x)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = torch.randn(4, 4)
        args = (x, y, z)

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            r".*BypassAOTAutogradCache: Unsupported call_function target .*",
        ):
            fn(*args)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 1)

        self._clear_dynamo_and_codecache()

        # TODO: Fix allow in graph
        raise unittest.SkipTest(
            "Allow in graph produces an unserializable cache artifact"
        )

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_fx_graph_cache_off(self):
        """
        Should not use cache if FXGraphCache is not enabled
        """

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # Clear FX graph cache: second call should also be a miss
        self._clear_dynamo_and_codecache()

        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    @dynamo_config.patch("compiled_autograd", True)
    def test_compiled_autograd_bypass(self):
        # Need to make the compiled autograd graph serializable
        def fn(a, b):
            out = a.cos() + b
            loss = out.sum()
            ga, gb = torch.autograd.grad(loss, inputs=[a, b])

        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="inductor")
        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "BypassAOTAutogradCache: Unsupported call_function target torch._dynamo.compiled_autograd.ops.validate_outputs",
        ):
            compiled_fn(a, b)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @dynamo_config.patch("compiled_autograd", True)
    def test_inference_graph_cache_hit_with_compiled_autograd_enabled(self):
        def fn(a, b):
            out = a.cos() + b
            return out.sum()

        a = torch.randn(25)
        b = torch.randn(25)
        compiled_fn = torch.compile(fn, backend="inductor")
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and run again. Should be a cache hit.
        counters.clear()
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_allow_custom_autograd_functions": True})
    def test_custom_autograd_function_miss(self):
        class MyAutogradFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x.sin()
                ctx.save_for_backward(y)
                ctx.foo = x.cos()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        def fn(a):
            return MyAutogradFunction.apply(a)

        a = torch.randn(5, device="cuda", requires_grad=True)
        a2 = a.clone().detach_().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        result.sum().backward()
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        class MyAutogradFunction(torch.autograd.Function):  # noqa: F811
            # Change the function slightly
            @staticmethod
            def forward(ctx, x):
                y = x.cos()
                ctx.save_for_backward(y)
                ctx.foo = x.sin()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        # Clear dynamo and run again. Should be a cache miss.
        counters.clear()
        self._clear_dynamo_and_codecache()
        result = compiled_fn(a2)
        self.assertEqual(fn(a2), result)
        result.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_allow_custom_autograd_functions": True})
    def test_custom_autograd_function(self):
        class MyAutogradFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x.sin()
                ctx.save_for_backward(y)
                ctx.foo = x.cos()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        def fn(a):
            return MyAutogradFunction.apply(a)

        a = torch.randn(5, device="cuda", requires_grad=True)
        a2 = a.clone().detach_().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        result.sum().backward()
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and run again. Should be a cache hit.
        counters.clear()
        self._clear_dynamo_and_codecache()
        result = compiled_fn(a2)
        self.assertEqual(fn(a2), result)
        result.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_allow_custom_autograd_functions": True})
    def test_custom_autograd_function_with_custom_triton_kernel(self):
        @triton.jit
        def my_jit(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            return y

        class MyAutogradFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = torch.ops.test.my_triton_op(x)
                ctx.save_for_backward(y)
                ctx.foo = x.cos()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        def fn(a):
            return MyAutogradFunction.apply(a)

        a = torch.randn(5, device=GPU_TYPE, requires_grad=True)
        a2 = a.clone().detach_().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)
        result.sum().backward()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and run again. Should be a cache hit.
        counters.clear()
        self._clear_dynamo_and_codecache()
        result = compiled_fn(a2)
        self.assertEqual(fn(a2), result)
        result.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_allow_custom_autograd_functions": True})
    def test_custom_autograd_function_with_custom_triton_kernel_cache_invalidation(
        self,
    ):
        @triton.jit
        def my_jit(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            return y

        class MyAutogradFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = torch.ops.test.my_triton_op(x)
                ctx.save_for_backward(y)
                ctx.foo = x.cos()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        def fn(a):
            return MyAutogradFunction.apply(a)

        a = torch.randn(5, device=GPU_TYPE, requires_grad=True)
        a2 = a.clone().detach_().requires_grad_(True)
        a3 = a.clone().detach_().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)
        result.sum().backward()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and run again. Should be a cache hit.
        counters.clear()
        self._clear_dynamo_and_codecache()
        result = compiled_fn(a2)
        self.assertEqual(fn(a2), result)
        result.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # Now modify the source code of my_jit by redefining it
        @triton.jit
        def my_jit(x):  # noqa: F811
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 2)  # Changed from +1 to +2

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            return y

        # Clear dynamo and run again. Should be a cache miss due to modified source code.
        counters.clear()
        self._clear_dynamo_and_codecache()
        compiled_fn = torch.compile(fn, backend="inductor")

        result = compiled_fn(a3)
        # Assert that after changing the source code, the cache no longer hits
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(fn(a3), result)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_cache_invalidation(self):
        from torch._library import capture_triton

        @triton.jit
        def my_jit(x):  # noqa: F811
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            y = x.clone().detach_().requires_grad_(True)
            capture_triton(my_jit)[1,](y)
            return y

        def fn(a):
            return torch.ops.test.my_triton_op(a)

        a = torch.randn(5, device=GPU_TYPE)
        a2 = a.clone().detach_()
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        self._clear_dynamo_and_codecache()

        # Redefine the triton op

        @triton.jit
        def my_jit(x):  # noqa: F811
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 2)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            return y

        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a2)

        # Second run should still miss
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        self.assertEqual(fn(a2), result)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_local_variable_kernel_detection(self):
        """
        Test that triton kernels passed via local variables are properly detected.

        This tests the pattern:
            kernel_fn = _my_kernel  # global
            wrapped = wrapper(kernel_fn)
            capture_triton(wrapped)[grid](...)

        The local variable tracing in get_inner_triton_kernels should trace
        through the assignments to find the original JITFunction.
        """
        from torch._library import capture_triton
        from torch._library.triton import triton_ops_to_kernels

        @triton.jit
        def inner_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(x_ptr + offsets, x + 42, mask=mask)

        def identity_wrapper(kernel):
            # simulate a wrapper function (like unroll_varargs in GDPA)
            # that takes a kernel as argument and returns it
            return kernel

        @torch._library.triton_op("test::local_var_triton_op", mutates_args=())
        def local_var_triton_op(x: torch.Tensor) -> torch.Tensor:
            y = x.clone()
            n_elements = y.numel()
            # this is the GDPA pattern:
            # 1. Assign global kernel to local variable
            # 2. Pass it through wrapper functions
            # 3. Use the wrapped result with capture_triton
            kernel_fn = inner_kernel  # Direct assignment from global
            wrapped_kernel = identity_wrapper(kernel_fn)  # Wrapper call
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            capture_triton(wrapped_kernel)[grid](y, n_elements, BLOCK_SIZE=256)
            return y

        kernels = triton_ops_to_kernels.get("test::local_var_triton_op", [])
        self.assertGreater(
            len(kernels),
            0,
            "Local variable tracing should detect the kernel",
        )

        kernel_names = [getattr(k, "__name__", str(k)) for k in kernels]
        self.assertIn(
            "inner_kernel",
            kernel_names,
            f"inner_kernel should be detected, got: {kernel_names}",
        )

        a = torch.randn(5, device=GPU_TYPE)
        expected = a.clone() + 42
        result = torch.ops.test.local_var_triton_op(a)
        self.assertEqual(result, expected)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_local_variable_cache_invalidation(self):
        """
        Test that cache properly invalidates when a kernel passed via local
        variable changes.
        """
        from torch._library import capture_triton

        @triton.jit
        def versioned_kernel(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::local_var_cache_test", mutates_args=())
        def local_var_cache_op(x: torch.Tensor) -> torch.Tensor:
            y = x.clone().detach_().requires_grad_(True)
            kernel = versioned_kernel  # local assignment
            capture_triton(kernel)[1,](y)
            return y

        def fn(a):
            return torch.ops.test.local_var_cache_test(a)

        a = torch.randn(5, device=GPU_TYPE)
        a2 = a.clone().detach_()
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        self._clear_dynamo_and_codecache()

        # redef the kernel with different behavior
        @triton.jit
        def versioned_kernel(x):  # noqa: F811
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 2)

        @torch._library.triton_op("test::local_var_cache_test", mutates_args=())
        def local_var_cache_op(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            y = x.clone().detach_().requires_grad_(True)
            kernel = versioned_kernel  # local assignment
            capture_triton(kernel)[1,](y)
            return y

        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a2)

        # Should be a cache miss due to kernel source change
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        self.assertEqual(fn(a2), result)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_recursive_function_kernel_detection(self):
        """
        Test that triton kernels hidden behind helper function calls are detected.

        This tests the recursive function analysis capability where:
            def helper():
                capture_triton(my_kernel)[grid](...)

            @triton_op(...)
            def my_op(x):
                helper()  # kernel is inside helper, not directly in my_op

        The recursive analysis in get_inner_triton_kernels should trace
        into helper functions to find the triton kernels.
        """
        from torch._library import capture_triton
        from torch._library.triton import triton_ops_to_kernels

        @triton.jit
        def nested_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(x_ptr + offsets, x + 100, mask=mask)

        def helper_that_calls_kernel(y, n_elements):
            """Helper function that contains the triton kernel call."""
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            capture_triton(nested_kernel)[grid](y, n_elements, BLOCK_SIZE=256)

        @torch._library.triton_op("test::recursive_func_triton_op", mutates_args=())
        def recursive_func_triton_op(x: torch.Tensor) -> torch.Tensor:
            y = x.clone()
            n_elements = y.numel()
            # The kernel is hidden inside helper_that_calls_kernel
            helper_that_calls_kernel(y, n_elements)
            return y

        kernels = triton_ops_to_kernels.get("test::recursive_func_triton_op", [])
        self.assertGreater(
            len(kernels),
            0,
            "Recursive function analysis should detect the kernel in helper function",
        )

        kernel_names = [getattr(k, "__name__", str(k)) for k in kernels]
        self.assertIn(
            "nested_kernel",
            kernel_names,
            f"nested_kernel should be detected, got: {kernel_names}",
        )

        a = torch.randn(5, device=GPU_TYPE)
        expected = a.clone() + 100
        result = torch.ops.test.recursive_func_triton_op(a)
        self.assertEqual(result, expected)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_kernel_factory_function(self):
        """
        Test that triton kernels returned from factory functions are properly detected.

        i.e., the pattern:
            kernel = get_autotune_kernel()
            capture_triton(kernel)[grid](...)
        """
        from torch._library import capture_triton
        from torch._library.triton import triton_ops_to_kernels

        @triton.jit
        def factory_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(x_ptr + offsets, x + 42, mask=mask)

        def get_kernel():
            """Factory function that returns a triton kernel."""
            return factory_kernel

        @torch._library.triton_op("test::factory_triton_op", mutates_args=())
        def factory_triton_op(x: torch.Tensor) -> torch.Tensor:
            y = x.clone()
            n_elements = y.numel()
            kernel = get_kernel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            capture_triton(kernel)[grid](y, n_elements, BLOCK_SIZE=256)
            return y

        kernels = triton_ops_to_kernels.get("test::factory_triton_op", [])

        self.assertGreater(
            len(kernels),
            0,
            "we should detect the kernel returned by get_kernel()",
        )

        kernel_names = [getattr(k, "__name__", str(k)) for k in kernels]
        self.assertIn(
            "factory_kernel",
            kernel_names,
            f"factory_kernel should be detected, got: {kernel_names}",
        )

        a = torch.randn(5, device=GPU_TYPE)
        expected = a.clone() + 42
        result = torch.ops.test.factory_triton_op(a)
        self.assertEqual(result, expected)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_lru_cache_factory_function(self):
        """
        Test that triton kernels returned from @lru_cache decorated factory
        functions are properly detected.

        i.e., the pattern:
            @lru_cache
            def get_autotune_kernel_lru():
                return get_autotune_kernel(...)

            kernel = get_autotune_kernel_lru()
            capture_triton(kernel)[grid](...)
        """
        from functools import lru_cache

        from torch._library import capture_triton
        from torch._library.triton import triton_ops_to_kernels

        @triton.jit
        def cached_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(x_ptr + offsets, x + 99, mask=mask)

        def get_kernel():
            """Factory function."""
            return cached_kernel

        @lru_cache
        def get_cached_kernel():
            """Calls factory function decorated with @lru_cache."""
            return get_kernel()

        @torch._library.triton_op("test::lru_cache_triton_op", mutates_args=())
        def lru_cache_triton_op(x: torch.Tensor) -> torch.Tensor:
            y = x.clone()
            n_elements = y.numel()
            kernel = get_cached_kernel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            capture_triton(kernel)[grid](y, n_elements, BLOCK_SIZE=256)
            return y

        kernels = triton_ops_to_kernels.get("test::lru_cache_triton_op", [])

        self.assertGreater(
            len(kernels),
            0,
            "we should detect the kernel through @lru_cache",
        )

        kernel_names = [getattr(k, "__name__", str(k)) for k in kernels]
        self.assertIn(
            "cached_kernel",
            kernel_names,
            f"cached_kernel should be detected, got: {kernel_names}",
        )

        a = torch.randn(5, device=GPU_TYPE)
        expected = a.clone() + 99
        result = torch.ops.test.lru_cache_triton_op(a)
        self.assertEqual(result, expected)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_cache_multiple_ops_invalidation(self):
        @triton.jit
        def my_jit(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @triton.jit
        def my_jit2(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            torch._library.capture_triton(my_jit2)[1,](y)
            return y

        @torch._library.triton_op("test::my_triton_op2", mutates_args=())
        def my_triton_op2(x: torch.Tensor) -> torch.Tensor:
            y = x.clone().detach_().requires_grad_(True)
            torch.ops.test.my_triton_op(y)
            return y

        def fn(a):
            return torch.ops.test.my_triton_op2(a)

        a = torch.randn(5, device=GPU_TYPE)
        a2 = a.clone().detach_()
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        self._clear_dynamo_and_codecache()

        # Redefine the triton op

        @triton.jit
        def my_jit(x):  # noqa: F811
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 2)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            torch._library.capture_triton(my_jit2)[1,](y)
            return y

        @torch._library.triton_op("test::my_triton_op2", mutates_args=())
        def my_triton_op2(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            y = x.clone().detach_().requires_grad_(True)
            torch.ops.test.my_triton_op(y)
            return y

        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a2)

        # Second run should still miss
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        self.assertEqual(fn(a2), result)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch({"fx_graph_cache": True})
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"strict_autograd_cache": True})
    def test_autograd_lazy_backward(self):
        """
        Lazily compile the backward, and lazily save to cache
        """

        def fn(a, b):
            return a.cos() + b

        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # Clear dynamo and run again. Should be a cache miss still, because backward hasn't run
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # Now let's run the backward
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and rerun everything, now there should be a cache hit
        self._clear_dynamo_and_codecache()
        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch({"fx_graph_cache": True})
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"strict_autograd_cache": True})
    def test_autograd_no_dynamo_trace_backward(self):
        """
        Test that dynamo does not trace into the backward compiled function,
        even on cache hit.
        """
        torch._dynamo.eval_frame.clear_dynamo_tls()

        @torch.compile
        def fn(x):
            # Calls x.sum().backward() during forward execution of fn
            (x_grad,) = torch.autograd.grad(x.sum(), x)
            return x_grad

        a = torch.randn(10, 10, requires_grad=True, device="cpu")
        result = fn(a)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        # Backward of `sum` will run during execution of graph break
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
        traced_frame_infos = copy.deepcopy(
            torch._dynamo.eval_frame.dynamo_tls.traced_frame_infos
        )

        torch._dynamo.reset()
        torch._dynamo.eval_frame.clear_dynamo_tls()
        result2 = fn(a)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
        new_traced_frame_infos = torch._dynamo.eval_frame.dynamo_tls.traced_frame_infos
        self.assertEqual(result, result2)
        # Dynamo should trace exactly the same frames on cache hit
        self.assertEqual(traced_frame_infos, new_traced_frame_infos)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_autograd_function(self):
        """
        Tests autograd cache hits
        """

        def fn(a, b):
            return a.sin() + b

        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Reset all tensors
        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @largeTensorTest("64GB", device=GPU_TYPE)
    @parametrize("device", (GPU_TYPE,))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_autograd_guard_single_entry(self, device, dtype):
        """
        Test caching the same graph, but under conditions that introduce guards
        for tensor sizes < int32. See test_codecache::TestFxGraphCache::test_cache_load_with_guards_int32_bounds.

        This test in particular tests the behavior of a single entry cache. If we ever make AOTAutogradCache
        support multiple entries under the same key, this test should be updated.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires CUDA SM80 or later")

        def fn(x, y):
            return (x + x, y + y)

        def expect_miss(compiled_fn, a, b):
            self._clear_dynamo_and_codecache()
            counters.clear()
            res = compiled_fn(a, b)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_guard_miss"],
                0,
            )
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            return res

        def expect_hit(compiled_fn, a, b):
            self._clear_dynamo_and_codecache()
            counters.clear()
            res = compiled_fn(a, b)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_guard_miss"],
                0,
            )
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_hit"],
                1,
            )
            return res

        def expect_guard_miss(compiled_fn, a, b):
            self._clear_dynamo_and_codecache()
            counters.clear()
            res = compiled_fn(a, b)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_guard_miss"],
                1,
            )
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_hit"],
                0,
            )
            return res

        compiled_fn = torch.compile(fn, dynamic=True)

        a_shape = (5, 6)
        b_shape = (7, 8)
        a = torch.rand(a_shape, device=device, dtype=dtype)
        b = torch.rand(b_shape, device=device, dtype=dtype)
        res1 = expect_miss(compiled_fn, a, b)

        # Same shape, should cache hit
        a2 = a.detach().clone()
        b2 = b.detach().clone()

        res2 = expect_hit(compiled_fn, a2, b2)

        self.assertEqual(res1, res2)

        # By changing the shape greatly, despite the same exact input
        # graph, inductor should report a guard miss, leading
        # to a cache miss on our end.
        a_shape = (5, 6)
        b_shape = (47000, 47001)
        a3 = torch.rand(a_shape, device=device, dtype=dtype)
        b3 = torch.rand(b_shape, device=device, dtype=dtype)

        expect_guard_miss(compiled_fn, a3, b3)

        # Wobble the shape a bit, but not enough
        # to trigger a guard miss (since 6, 7 is still less than int32)
        # Should result in a cache hit
        a_shape = (6, 7)
        b_shape = (47000, 47001)
        a4 = torch.rand(a_shape, device=device, dtype=dtype)
        b4 = torch.rand(b_shape, device=device, dtype=dtype)
        expect_hit(compiled_fn, a4, b4)

        # Change the shape back to the original,
        # FXGraphCache should hit because it stores
        # multiple entries
        a_shape = (5, 6)
        b_shape = (7, 8)
        a5 = torch.rand(a_shape, device=device, dtype=dtype)
        b5 = torch.rand(b_shape, device=device, dtype=dtype)
        expect_hit(compiled_fn, a5, b5)

    @largeTensorTest("64GB", device=GPU_TYPE)
    @parametrize("device", (GPU_TYPE,))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize("requires_grad", (True, False))
    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_autograd_inductor_guards(self, device, dtype, requires_grad):
        """
        Test caching the same graph, but under conditions that introduce guards
        for tensor sizes < int32.
        See test_codecache::TestFxGraphCache::test_cache_load_with_guards_int32_bounds.
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
        expected_hits = expected_misses = expected_saves = 0
        expected_guard_misses = 0
        for a_shape, b_shape in shapes:
            a = torch.rand(
                a_shape, device=device, dtype=dtype, requires_grad=requires_grad
            )
            b = torch.rand(
                b_shape, device=device, dtype=dtype, requires_grad=requires_grad
            )

            # AVOID a dynamo reset here. We expect guards to have been
            # added that will be violated with the new shape. We should
            # see a recompilation (along with a cache miss).
            res1 = compiled_fn(a, b)
            # A first call should miss in the cache.
            expected_misses += 1  # noqa: SIM113
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_miss"], expected_misses
            )
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_guard_miss"],
                expected_guard_misses,
            )

            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_hit"], expected_hits
            )
            # Because dynamic shapes are enabled, we expect backwards to be compiled ahead of time
            # So we should see a cache save here
            expected_saves += 1  # noqa: SIM113
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_saved"], expected_saves
            )
            if requires_grad:
                res1[0].sum().backward()
                # No extra saves
                self.assertEqual(
                    counters["aot_autograd"]["autograd_cache_saved"], expected_saves
                )

            a2 = a.detach().clone().requires_grad_(requires_grad)
            b2 = b.detach().clone().requires_grad_(requires_grad)
            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).

            # Now clear dynamo and we should see a cache hit
            # This should populate guards to dynamo's cache, so that a subsequent run with a different
            # shape will still trigger a second call to autograd_cache.
            self._clear_dynamo_and_codecache()
            res2 = compiled_fn(a2, b2)
            expected_hits += 1  # noqa: SIM113
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_miss"], expected_misses
            )
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_guard_miss"],
                expected_guard_misses,
            )
            # First compile is a regular cache miss, subsequent are guard misses
            expected_guard_misses += 1  # noqa: SIM113
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_hit"], expected_hits
            )
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_saved"], expected_saves
            )
            self.assertEqual(res1, res2)
            if requires_grad:
                res2[0].sum().backward()
                self.assertEqual(a.grad, a2.grad)

    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_nn_module_with_params_global_constant(self):
        class MyMod(torch.nn.Module):
            CONSTANT = torch.tensor([[2, 2], [2, 2]])

            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn([2, 2]))

            def forward(self, x):
                return x.sin() + self.param + MyMod.CONSTANT

        with torch.no_grad():
            compiled_fn = torch.compile(MyMod(), backend="inductor", fullgraph=True)
            res1 = compiled_fn(torch.ones([2, 2]))
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

            self._clear_dynamo_and_codecache()
            res2 = compiled_fn(torch.ones([2, 2]))
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

            self.assertEqual(res1, res2)
            # Edit the "constant". We'll get a cache hit,
            # but it should result in a different result when run
            # because MyMod.CONSTANT is an input to the graph
            MyMod.CONSTANT = torch.tensor([[3, 3], [3, 3]])
            self._clear_dynamo_and_codecache()
            res3 = compiled_fn(torch.ones([2, 2]))
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
            self.assertNotEqual(res1, res3)
            self.assertEqual(res1, res3.sub(torch.ones(2, 2)))

    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_constant_tensor_device_guards(self):
        """
        Usually, when there are example inputs, the device index of the inputs
        is sufficient to make sure we don't cache hit with the results from different
        cuda devices.
        When the input has no arguments, we still need to have the cuda
        device index in the cache key.
        """

        @torch.compile(backend="inductor")
        def f():
            y = torch.tensor([5], device="cuda")
            return (y,)

        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            result = f()
            self.assertEqual(result[0].device, torch.device("cuda:0"))

        self._clear_dynamo_and_codecache()

        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            result = f()
            self.assertEqual(result[0].device, torch.device("cuda:1"))

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_multiple_compile_triton_kernels(self):
        """
        When we cache hit on AOTAutogradCache, we need to still clear
        CompiledTritonKernels after compiling the kernel.
        """
        from torch._inductor.async_compile import CompiledTritonKernels

        @torch.compile
        def f(x, y):
            return x.sin() + y

        x = torch.randn(10, device="cuda")
        y = torch.randn(10, device="cuda")
        with torch.no_grad():
            result = f(x, y)
            self.assertEqual(result, x.sin() + y)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(len(CompiledTritonKernels._cache), 0)

        self._clear_dynamo_and_codecache()
        with torch.no_grad():
            result = f(x, y)
            self.assertEqual(result, x.sin() + y)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(len(CompiledTritonKernels._cache), 0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"strict_autograd_cache": True})
    def test_dynamic_shapes_different_sizes(self):
        # The forward and backward function have different symint inputs,
        # but the same underlying symbols
        def fn(x, y):
            z = x * y
            return (torch.cat((x, x), dim=0), z)

        (x1, y1) = torch.randn(5, requires_grad=True), torch.randn(5)
        compiled_fn = torch.compile(fn, backend="inductor", dynamic=True)
        x_compiled, _ = compiled_fn(x1, y1)
        x_compiled.sum().backward()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
        self._clear_dynamo_and_codecache()

        # Run a second time and see it cache hit instead of erroring
        (x2, y2) = torch.randn(5, requires_grad=True), torch.randn(5)
        x_compiled, _ = compiled_fn(x2, y2)
        x_compiled.sum().backward()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not SM80OrLater, "bfloat16, float8")
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"activation_memory_budget": 1.0})
    @functorch_config.patch({"activation_memory_budget_runtime_estimator": "testing"})
    @functorch_config.patch({"saved_tensors_hooks_filtering_mode": "all"})
    def test_saved_tensors_hooks_autograd_cache(self):
        ctx = torch.autograd.graph.saved_tensors_hooks
        device = torch.device("cuda:0")

        def pack_cpu(x):
            return x.to(device="cpu")

        def unpack_cpu(x):
            return x.to(device=device)

        def pack_cpu2(x):
            return x.to(device="cpu")

        def unpack_cpu2(x):
            return x.to(device=device)

        def pack_mul2(x):
            return x * 2

        def unpack_mul2(x):
            return x / 2

        # Can not use custom AutogradFunction here,
        # Cache bypasses AutogradFunction Ctx usage.
        # Can not save in ctx non floating point dtypes.
        # For non-symbolic tracing all dtypes and devices and burned in the graph.

        def fn(x):
            x = x + 1
            x = x.sin().cos()
            x = x.relu()
            x = x.exp()
            x = 2 * x
            return x

        backend = "inductor"

        def inp_fn():
            x = torch.ones(2, 3, device=device, requires_grad=True)
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(x, 1)
            return x

        x = inp_fn()
        fn_compiled = torch.compile(fn, backend=backend, fullgraph=True)
        y = fn_compiled(x)
        y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        with ctx(
            *saved_tensors_hooks_to_gm(
                pack_cpu,
                unpack_cpu,
                symbolic_tracing=False,
                inp_fn=inp_fn,
                pack_cache_hash="cpu_offload",
                unpack_cache_hash="cpu_offload",
            )
        ):
            x = inp_fn()
            y = fn_compiled(x)
            y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        with ctx(
            *saved_tensors_hooks_to_gm(
                pack_cpu2,
                unpack_cpu2,
                symbolic_tracing=False,
                inp_fn=inp_fn,
                pack_cache_hash="cpu_offload",
                unpack_cache_hash="cpu_offload",
            )
        ):
            x = inp_fn()
            y = fn_compiled(x)
            y.sum().backward()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        with ctx(
            *saved_tensors_hooks_to_gm(pack_mul2, unpack_mul2, symbolic_tracing=False)
        ):
            x = inp_fn()
            y = fn_compiled(x)
            y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 3)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 3)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not SM80OrLater, "bfloat16, float8")
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_saved_tensors_hooks_autograd_cache_symbolic(self):
        def pack_fp8_with_scale(x):
            return _pack_fp8_with_scale_wrap(x)

        def unpack_fp8_with_scale(packed):
            return _unpack_fp8_with_scale_wrap(packed)

        ctx = torch.autograd.graph.saved_tensors_hooks

        def fn(x):
            x = x + 1
            # Relu saves bitmask in AutogradContext
            x = x.relu()
            x = x.relu()
            return x

        device = torch.device("cuda:0")
        backend = "inductor"

        def inp_fn():
            x = torch.ones(2, 3, device=device, requires_grad=True)
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(x, 1)
            return x

        x = inp_fn()
        fn_compiled = torch.compile(fn, backend=backend, fullgraph=True)
        y = fn_compiled(x)
        y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        with ctx(
            *saved_tensors_hooks_to_gm(
                pack_fp8_with_scale,
                unpack_fp8_with_scale,
                "fp8_with_scale_dtype_floating_point",
                "fp8_with_scale_dtype_floating_point",
            )
        ):
            x = inp_fn()
            y = fn_compiled(x)
            y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        with ctx(
            *saved_tensors_hooks_to_gm(
                pack_fp8_with_scale,
                unpack_fp8_with_scale,
                "fp8_with_scale_dtype_floating_point",
                "fp8_with_scale_dtype_floating_point",
            )
        ):
            x = inp_fn()
            y = fn_compiled(x)
            y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        with ctx(
            *saved_tensors_hooks_to_gm(
                pack_fp8_with_scale,
                unpack_fp8_with_scale,
                "fp8_with_scale_dtype_floating_point_MISS",
                "fp8_with_scale_dtype_floating_point_MISS",
            )
        ):
            x = inp_fn()
            y = fn_compiled(x)
            y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 3)

    @functorch_config.patch({"enable_autograd_cache": True})
    @inductor_config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "autotune_local_cache": True,
        }
    )
    def test_cache_lazy_backward_for_compiled_autograd(self):
        device = "cpu"
        dtype = torch.float32
        dynamic = True
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
            expected_grads = torch.autograd.grad(eager_result.sum(), inputs=(a, b))
            compiled_result = compiled_fn(a, b)
            with torch._dynamo.compiled_autograd._enable(
                torch.compile(dynamic=dynamic)
            ):
                actual_grads = torch.autograd.grad(compiled_result.sum(), inputs=(a, b))
            if hasattr(a, "_dynamo_weak_dynamic_indices"):
                del a._dynamo_weak_dynamic_indices
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(expected_grads[0], actual_grads[0])
            self.assertEqual(expected_grads[1], actual_grads[1])
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            else:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 3)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
            self.assertEqual(counters["compiled_autograd"]["captures"], 1)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        autotune_expect = 2 if device == GPU_TYPE else 0

        if functorch_config.bundled_autograd_cache:
            self.assertEqual(len(cache_info.inductor_artifacts), 0)
        else:
            self.assertEqual(len(cache_info.inductor_artifacts), 3)
        self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
        self.assertEqual(len(cache_info.pgo_artifacts), 0)

        self._clear_all_caches()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit, should not recompile
        with fresh_cache():
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)

            if functorch_config.bundled_autograd_cache:
                self.assertEqual(len(cache_info.inductor_artifacts), 0)
            else:
                self.assertEqual(len(cache_info.inductor_artifacts), 3)
            self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
            self.assertEqual(len(cache_info.pgo_artifacts), 0)

            for i in range(3):
                counters.clear()
                eager_result = fn(a, b)
                expected_grads = torch.autograd.grad(eager_result.sum(), inputs=(a, b))
                compiled_result = compiled_fn(a, b)
                with torch._dynamo.compiled_autograd._enable(
                    torch.compile(dynamic=dynamic)
                ):
                    actual_grads = torch.autograd.grad(
                        compiled_result.sum(), inputs=(a, b)
                    )
                if hasattr(a, "_dynamo_weak_dynamic_indices"):
                    del a._dynamo_weak_dynamic_indices
                self.assertEqual(eager_result, compiled_result)
                self.assertEqual(expected_grads[0], actual_grads[0])
                self.assertEqual(expected_grads[1], actual_grads[1])

                if i == 0:
                    # initial compile
                    if functorch_config.bundled_autograd_cache:
                        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                    else:
                        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 3)
                        self.assertEqual(
                            counters["inductor"]["fxgraph_lookup_write_file"], 3
                        )
                    self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
                    self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
                    self.assertEqual(
                        counters["aot_autograd"]["autograd_cache_saved"], 0
                    )
                    self.assertEqual(counters["compiled_autograd"]["captures"], 1)
                else:
                    # no recompiles
                    self.assertFalse(counters)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"bundled_autograd_cache": True})
    def test_regional_inductor_basic(self):
        """
        Basic test for regional inductor with bundled autograd cache.
        Tests that regional inductor compilation results can be cached and hit.
        """
        import torch.fx.traceback as fx_traceback

        def fn(x, y):
            sin = torch.sin(x)
            # Mark this region to be compiled with inductor
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1
            return torch.sin(add)

        x = torch.randn(10, device="cpu")
        y = torch.randn(10, device="cpu")

        # Compile with regional inductor backend
        compiled_fn = torch.compile(
            fn, backend=aot_eager_regional_inductor(), fullgraph=True
        )

        # First call should miss in cache
        result1 = compiled_fn(x, y)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Second call should hit (after clearing dynamo)
        self._clear_dynamo_and_codecache()
        result2 = compiled_fn(x, y)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Results should be the same
        self.assertEqual(result1, result2)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"bundled_autograd_cache": True})
    def test_regional_inductor_with_backward(self):
        """
        Test regional inductor with backward pass and bundled autograd cache.
        Note: Regional inductor triggers multiple AOT autograd compilations:
        - One for the outer graph (with regional inductor backend)
        - One for each marked region (via standalone_compile)
        """
        import torch.fx.traceback as fx_traceback

        def fn(x, y):
            sin = torch.sin(x)
            # Mark this region to be compiled with inductor
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1
            return torch.sin(add)

        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        y2 = y.detach().clone().requires_grad_(True)

        # Compile with regional inductor backend
        compiled_fn = torch.compile(
            fn, backend=aot_eager_regional_inductor(), fullgraph=True
        )

        # First call: AOT autograd compiles the outer graph (1 miss)
        # Regional inductor then compiles the marked region (1 more miss)
        result1 = compiled_fn(x, y)
        result1.sum().backward()

        # We expect 2 cache misses: outer graph + marked region
        initial_misses = counters["aot_autograd"]["autograd_cache_miss"]
        initial_saves = counters["aot_autograd"]["autograd_cache_saved"]
        self.assertGreater(initial_misses, 0)
        self.assertGreater(initial_saves, 0)

        # Second call should hit (after clearing dynamo)
        self._clear_dynamo_and_codecache()
        result2 = compiled_fn(x2, y2)
        result2.sum().backward()

        # Should have cache hits now
        final_hits = counters["aot_autograd"]["autograd_cache_hit"]
        self.assertGreater(final_hits, 0)

        # Cache misses and saves should not increase
        self.assertEqual(
            counters["aot_autograd"]["autograd_cache_miss"], initial_misses
        )
        self.assertEqual(
            counters["aot_autograd"]["autograd_cache_saved"], initial_saves
        )

        # Results and gradients should be the same
        self.assertEqual(result1, result2)
        self.assertEqual(x.grad, x2.grad)
        self.assertEqual(y.grad, y2.grad)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"bundled_autograd_cache": True})
    def test_regional_inductor_cache_miss_on_change(self):
        """
        Test that changing the function causes a cache miss with regional inductor.
        Regional inductor creates multiple AOT compilations, so we track
        the change in cache misses rather than absolute counts.
        """
        import torch.fx.traceback as fx_traceback

        def fn1(x, y):
            sin = torch.sin(x)
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1
            return torch.sin(add)

        def fn2(x, y):
            sin = torch.sin(x)
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 2  # Changed from +1 to +2
            return torch.sin(add)

        x = torch.randn(10)
        y = torch.randn(10)

        # Compile first function
        compiled_fn1 = torch.compile(
            fn1, backend=aot_eager_regional_inductor(), fullgraph=True
        )
        result1 = compiled_fn1(x, y)
        first_misses = counters["aot_autograd"]["autograd_cache_miss"]
        first_saves = counters["aot_autograd"]["autograd_cache_saved"]
        self.assertGreater(first_misses, 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertGreater(first_saves, 0)

        # Compile second function (different graph)
        self._clear_dynamo_and_codecache()
        compiled_fn2 = torch.compile(
            fn2, backend=aot_eager_regional_inductor(), fullgraph=True
        )
        result2 = compiled_fn2(x, y)
        # Should miss because graph is different (more misses than before)
        self.assertGreater(
            counters["aot_autograd"]["autograd_cache_miss"], first_misses
        )
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertGreater(
            counters["aot_autograd"]["autograd_cache_saved"], first_saves
        )

        # Results should be different
        self.assertNotEqual(result1, result2)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"activation_memory_budget": 0.5})
    def test_custom_runtime_estimator_cache_hit(self):
        """Test that CustomRuntimeEstimator with valid uuid() results in cache hit."""

        class MyCustomEstimator(CustomRuntimeEstimator):
            def __call__(self, node) -> float:
                return 1.0

            def uuid(self):
                return "my_custom_estimator_v1"

        def fn(x, y):
            return torch.mm(x, y) + 1

        with fresh_cache():
            with functorch_config.patch(
                {"activation_memory_budget_runtime_estimator": MyCustomEstimator()}
            ):
                compiled_fn = torch.compile(fn, backend="inductor")

                x = torch.randn(10, 10, requires_grad=True)
                y = torch.randn(10, 10, requires_grad=True)
                result = compiled_fn(x, y)
                result.sum().backward()

                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)

                self._clear_dynamo_and_codecache()

                x2 = torch.randn(10, 10, requires_grad=True)
                y2 = torch.randn(10, 10, requires_grad=True)
                result2 = compiled_fn(x2, y2)
                result2.sum().backward()

                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"activation_memory_budget": 0.5})
    def test_custom_estimator_bypasses_cache(self):
        """Test that cache is bypassed when custom estimator lacks proper uuid."""

        def fn(x, y):
            return torch.mm(x, y) + 1

        # Test 1: Raw callable without uuid() method bypasses cache
        def raw_estimator(node) -> float:
            return 1.0

        with fresh_cache():
            with functorch_config.patch(
                {"activation_memory_budget_runtime_estimator": raw_estimator}
            ):
                compiled_fn = torch.compile(fn, backend="inductor")
                x = torch.randn(10, 10, requires_grad=True)
                y = torch.randn(10, 10, requires_grad=True)
                result = compiled_fn(x, y)
                result.sum().backward()

                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)
                self.assertGreater(counters["aot_autograd"]["autograd_cache_bypass"], 0)

        counters.clear()
        self._clear_dynamo_and_codecache()

        # Test 2: CustomRuntimeEstimator with uuid() returning None bypasses cache
        class EstimatorWithNullUuid(CustomRuntimeEstimator):
            def __call__(self, node) -> float:
                return 1.0

            def uuid(self):
                return None

        with fresh_cache():
            with functorch_config.patch(
                {"activation_memory_budget_runtime_estimator": EstimatorWithNullUuid()}
            ):
                compiled_fn = torch.compile(fn, backend="inductor")
                x = torch.randn(10, 10, requires_grad=True)
                y = torch.randn(10, 10, requires_grad=True)
                result = compiled_fn(x, y)
                result.sum().backward()

                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)
                self.assertGreater(counters["aot_autograd"]["autograd_cache_bypass"], 0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"activation_memory_budget": 0.5})
    def test_different_custom_estimator_uuids_cause_cache_miss(self):
        """Test that different uuid values cause cache miss."""

        class EstimatorV1(CustomRuntimeEstimator):
            def __call__(self, node) -> float:
                return 1.0

            def uuid(self):
                return "estimator_v1"

        class EstimatorV2(CustomRuntimeEstimator):
            def __call__(self, node) -> float:
                return 1.0

            def uuid(self):
                return "estimator_v2"

        def fn(x, y):
            return torch.mm(x, y) + 1

        with fresh_cache():
            with functorch_config.patch(
                {"activation_memory_budget_runtime_estimator": EstimatorV1()}
            ):
                compiled_fn = torch.compile(fn, backend="inductor")
                x = torch.randn(10, 10, requires_grad=True)
                y = torch.randn(10, 10, requires_grad=True)
                result = compiled_fn(x, y)
                result.sum().backward()

                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)

            self._clear_dynamo_and_codecache()

            with functorch_config.patch(
                {"activation_memory_budget_runtime_estimator": EstimatorV2()}
            ):
                compiled_fn2 = torch.compile(fn, backend="inductor")
                x2 = torch.randn(10, 10, requires_grad=True)
                y2 = torch.randn(10, 10, requires_grad=True)
                result2 = compiled_fn2(x2, y2)
                result2.sum().backward()

                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)

    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"strict_autograd_cache": True})
    def test_output_views_input_dynamic(self):
        class OutputViewsInput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, x):
                batch, seq, hidden = x.shape
                y = self.linear(x)
                y = torch.relu(y)
                x_view = x.view(batch * seq, hidden)
                return y, x_view

        model = OutputViewsInput()
        x = torch.randn(2, 16, 64)

        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)

        compiled = torch.compile(model, backend="inductor", dynamic=True)

        # First call - should miss
        y, x_view = compiled(x)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Reset dynamo but keep cache
        self._clear_dynamo_and_codecache()

        # Use different dynamic shape
        x2 = torch.randn(4, 8, 64)
        torch._dynamo.mark_dynamic(x2, 0)
        torch._dynamo.mark_dynamic(x2, 1)

        # Second call with different shape - should hit cache
        y2, x_view2 = compiled(x2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Verify correctness
        eager_model = copy.deepcopy(model)
        expected_y, expected_view = eager_model(x2)
        self.assertEqual(y2, expected_y)
        self.assertEqual(x_view2, expected_view)
        # Verify the view shares storage with input (critical for view correctness)
        self.assertEqual(
            x_view2.untyped_storage().data_ptr(), x2.untyped_storage().data_ptr()
        )


@functorch_config.patch({"bundled_autograd_cache": True})
class AOTAutogradCacheBundledTests(AOTAutogradCacheTests):
    pass


@dataclasses.dataclass
class _MockEntryForPickleTest:
    """Module-level mock entry for pickle tests, so pickle errors come from fields not the class."""

    picklable_field: str
    unpicklable_field: object

    def pre_save(self):
        pass


@inductor_config.patch("fx_graph_cache", True)
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
            precompile_backend_id=None,
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
        shape_env = ShapeEnv()
        ctx = TracingContext(FakeTensorMode(shape_env=shape_env))
        # Needs a shape env for FxGraphCache.check_can_cache to pass.
        # Not needed for actual key calculation.
        with torch._guards.tracing(ctx):
            with sanitize_gm_for_cache(fx_g):
                return autograd_cache_key(fx_g, example_inputs, config, {})

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

        def fn2(x):  # noqa: F841
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

    def test_private_namespace(self):
        # TODO: anyone who monkeypatches a **public** function into torch namespace with @allow_in_graph
        # could still break our sanity check and cache something bad. But that's an edge case we'll take the risk on.
        # Monkeypatch some random private function into torch, see that it fails
        @torch._dynamo.allow_in_graph
        def my_private_fun(x):
            return x.sin()

        with patch("torch._my_priv", new=my_private_fun, create=True):

            def fn(x):
                return torch._my_priv(x)

            config = self.default_config()
            self.assertRaises(
                BypassAOTAutogradCache, lambda: self.gen_cache_key(fn, config)
            )

    @torch._inductor.config.patch({"freezing": True})
    def test_freezing(self):
        def fn(x):
            return x.cos().sin()

        config = self.default_config()
        self.assertRaises(
            BypassAOTAutogradCache, lambda: self.gen_cache_key(fn, config)
        )

    def test_private_builtin(self):
        # _foreach_add is a private torch function, but
        # it's also a builtin_function_or_method, so it should be allowed to be cached
        # since dynamo allows it in the graph
        def fn(x, b):
            y = (x, x)
            return torch._foreach_add(y, b)

        config = self.default_config()
        r1 = self.gen_cache_key(fn, config, inputs=[torch.ones(3), 1])
        r2 = self.gen_cache_key(fn, config, inputs=[torch.ones(3), 2])
        self.assertNotEqual(r1, r2)

    def test_nn_module_with_params(self):
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seq = torch.nn.Parameter(torch.ones((3, 3)))

            def forward(self, x):
                return self.seq + x

        config = self.default_config()
        # Different inputs and parameters, but all the same size
        c1 = self.gen_cache_key(MyMod(), config, inputs=[torch.ones((3, 3))])
        c2 = self.gen_cache_key(MyMod(), config, inputs=[torch.ones((3, 3))])
        self.assertEqual(c1, c2)

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

    def test_safe_torchfunction(self):
        def fn(x):
            a = x.size()
            b = torch.Size([3, 3])
            c = a == b
            x = torch.sym_int(9)
            y = torch.sym_float(x)
            z = torch.sym_int(torch.sym_sqrt(y))
            result = torch.sym_sum([x, y, z])
            return (c, result)

        config = self.default_config()
        self.gen_cache_key(fn, config, inputs=[torch.ones((3, 3))])

    def test_sanitize_gm_for_cache(self):
        def fn(x):
            y = torch.sin(x)
            z = torch.cos(x)
            w = y + z
            w.abs()
            return w

        _, fx_g, example_inputs = self._get_dynamo_output(fn, torch.ones(3))

        ctx = TracingContext(FakeTensorMode(shape_env=ShapeEnv()))
        with torch._guards.tracing(ctx):
            fx_g.meta = {"foo": "bar"}
            fx_g.compile_subgraph_reason = "Blah"
            config = self.default_config()
            with sanitize_gm_for_cache(fx_g):
                c1 = autograd_cache_key(fx_g, example_inputs, config, {})
            c3 = autograd_cache_key(fx_g, example_inputs, config, {})

            fx_g.meta = {"foo": "baz"}
            fx_g.compile_subgraph_reason = None
            with sanitize_gm_for_cache(fx_g):
                c2 = autograd_cache_key(fx_g, example_inputs, config, {})
            c4 = autograd_cache_key(fx_g, example_inputs, config, {})

            self.assertEqual(c1, c2)
            self.assertNotEqual(c3, c4)

    def test_dill_serialization_with_inner_functions(self):
        """
        Test that with dill, we can now serialize graphs that contain inner functions
        in node metadata. Standard pickle would fail on these because inner functions
        are not defined at module level, but dill serializes function code objects and
        closures directly.
        """
        from torch.fx._graph_pickler import GraphPickler, Options
        from torch.utils._import_utils import import_dill

        dill = import_dill()
        if dill is None:
            self.skipTest("dill not available")

        def fn(x):
            return x.sin().cos()

        _, fx_g, example_inputs = self._get_dynamo_output(fn, torch.ones(3))

        def inner_helper(x):
            return x * 2

        for node in fx_g.graph.nodes:
            node.meta["inner_fn"] = inner_helper

        options = Options(node_metadata_key_filter=None)
        serialized = GraphPickler.dumps(fx_g, options)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        deserialized = GraphPickler.loads(serialized, fake_mode)

        self.assertIsInstance(deserialized, torch.fx.GraphModule)
        for node in deserialized.graph.nodes:
            self.assertIn("inner_fn", node.meta)
            self.assertEqual(node.meta["inner_fn"](5), 10)

    def test_pickle_entry_with_unpicklable_field(self):
        from weakref import WeakValueDictionary

        # WeakValueDictionary's internal 'remove' callback is unpicklable
        weak_dict = WeakValueDictionary()
        entry = _MockEntryForPickleTest(
            picklable_field="test",
            unpicklable_field=weak_dict,
        )

        with self.assertLogs(
            "torch._functorch._aot_autograd.autograd_cache", level="WARNING"
        ) as log_context:
            result = AOTAutogradCache._pickle_entry(entry, remote=False)

        self.assertIsNone(result)
        self.assertEqual(len(log_context.output), 1)
        self.assertIn(
            "WeakValueDictionary.__init__.<locals>.remove",  # noqa: B950
            log_context.output[0],
        )

    def test_pickle_entry_with_lambda(self):
        entry = _MockEntryForPickleTest(
            picklable_field="test",
            unpicklable_field=lambda x: x,  # lambdas are unpicklable
        )

        with (
            patch("torch._logging.trace_structured") as mock_trace,
            self.assertLogs(
                "torch._functorch._aot_autograd.autograd_cache", level="WARNING"
            ) as log_context,
        ):
            result = AOTAutogradCache._pickle_entry(entry, remote=False)

        self.assertIsNone(result)
        self.assertEqual(len(log_context.output), 1)
        self.assertIn(
            """AOTAutogradCachePicklerTests.test_pickle_entry_with_lambda.<locals>.<lambda>""",  # noqa: B950
            log_context.output[0],
        )
        mock_trace.assert_called_once()
        call_args = mock_trace.call_args
        self.assertEqual(call_args[0][0], "artifact")
        metadata = call_args[1]["metadata_fn"]()
        self.assertEqual(metadata["name"], "aotautograd_cache_pickle_failure")
        self.assertEqual(metadata["encoding"], "json")

    @functorch_config.patch("strict_autograd_cache", True)
    def test_pickle_entry_strict_mode_raises(self):
        entry = _MockEntryForPickleTest(
            picklable_field="test",
            unpicklable_field=lambda x: x,
        )

        # Python 3.14+ raises PicklingError with "Can't pickle local object"
        # Earlier versions raise AttributeError with "Can't get local object"
        with self.assertRaisesRegex(
            (AttributeError, pickle.PicklingError),
            r"AOTAutogradCachePicklerTests.test_pickle_entry_strict_mode_raises.<locals>.<lambda>",
        ):
            AOTAutogradCache._pickle_entry(entry, remote=False)


def _policy_save_mm(ctx, op, *args, **kwargs):
    if op == torch.ops.aten.mm.default:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_RECOMPUTE


def _policy_save_add(ctx, op, *args, **kwargs):
    if op == torch.ops.aten.add.Tensor:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_RECOMPUTE


def _policy_no_hash(ctx, op, *args, **kwargs):
    if op == torch.ops.aten.mm.default:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_RECOMPUTE


def _create_sac_ctx_fn(policy, cache_hash=None):
    """
    Helper to create a SAC context_fn with cache_hash set on the partial.
    """
    ctx_fn = functools.partial(create_selective_checkpoint_contexts, policy)
    if cache_hash is not None:
        ctx_fn.cache_hash = cache_hash
    return ctx_fn


class HOPCacheTests(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        counters.clear()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        counters.clear()
        torch._dynamo.reset()

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_different_sac_policies_cause_cache_miss(self):
        def gn(x, y):
            a = torch.mm(x, y)
            b = torch.add(a, x)
            return b

        ctx_fn_save_mm = _create_sac_ctx_fn(_policy_save_mm, "policy_save_mm_v1")

        ctx_fn_save_add = _create_sac_ctx_fn(_policy_save_add, "policy_save_add_v1")

        @torch.compile(backend="inductor")
        def fn_with_checkpoint(x, y, ctx_fn):
            return checkpoint(gn, x, y, use_reentrant=False, context_fn=ctx_fn)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        with fresh_cache():
            fn_with_checkpoint(x, y, ctx_fn_save_mm)

            # First run: miss=1, hit=0
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            torch._dynamo.reset()

            fn_with_checkpoint(x, y, ctx_fn_save_add)

            # Different policy: miss=2, hit=0 (policy is part of cache key)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_same_sac_policy_causes_cache_hit(self):
        def gn(x, y):
            a = torch.mm(x, y)
            b = torch.add(a, x)
            return b

        ctx_fn = _create_sac_ctx_fn(_policy_save_mm, "policy_save_mm_v1")

        @torch.compile(backend="inductor")
        def fn_with_checkpoint(x, y):
            return checkpoint(gn, x, y, use_reentrant=False, context_fn=ctx_fn)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        with fresh_cache():
            fn_with_checkpoint(x, y)

            # First run: miss=1, hit=0
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            torch._dynamo.reset()

            fn_with_checkpoint(x, y)

            # Same policy: miss stays at 1, hit increments to 1
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_checkpoint_with_dropout_caches_correctly(self):
        def gn(x):
            return torch.dropout(x, p=0.5, train=True)

        @torch.compile(backend="inductor")
        def fn(x):
            return checkpoint(gn, x, use_reentrant=False)

        x = torch.randn(4, 4, requires_grad=True)

        with fresh_cache():
            out1 = fn(x.clone().requires_grad_(True))
            out1.sum().backward()

            # First run: miss=1, hit=0
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            torch._dynamo.reset()

            out2 = fn(x.clone().requires_grad_(True))
            out2.sum().backward()

            # Same function with RNG HOPs: miss stays at 1, hit increments to 1
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch("enable_autograd_cache", True)
    def test_sac_without_cache_hash_bypasses_cache(self):
        def gn(x, y):
            a = torch.mm(x, y)
            b = torch.add(a, x)
            return b

        ctx_fn = _create_sac_ctx_fn(_policy_no_hash)  # No cache_hash

        @torch.compile(backend="inductor")
        def fn_with_checkpoint(x, y):
            return checkpoint(gn, x, y, use_reentrant=False, context_fn=ctx_fn)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        with fresh_cache():
            fn_with_checkpoint(x, y)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertGreater(counters["aot_autograd"]["autograd_cache_bypass"], 0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_changing_cache_hash_invalidates_cache(self):
        def gn(x, y):
            a = torch.mm(x, y)
            b = torch.add(a, x)
            return b

        ctx_fn_mm = _create_sac_ctx_fn(_policy_save_mm, "policy_save_mm_v1")

        ctx_fn_add = _create_sac_ctx_fn(_policy_save_add, "policy_save_add_v1")

        @torch.compile(backend="inductor")
        def fn_mm(x, y):
            return checkpoint(gn, x, y, use_reentrant=False, context_fn=ctx_fn_mm)

        @torch.compile(backend="inductor")
        def fn_add(x, y):
            return checkpoint(gn, x, y, use_reentrant=False, context_fn=ctx_fn_add)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        with fresh_cache():
            fn_mm(x, y)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            torch._dynamo.reset()

            fn_add(x, y)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
