# Owner(s): ["module: inductor"]


import threading
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
from torch._higher_order_ops import flex_gemm
from torch._inductor import config
from torch._inductor.codegen.cuda.cuda_env import is_datacenter_blackwell_arch
from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_scheduling import (
    EPILOGUE_FN_NAME,
)
from torch._inductor.heuristics.template.nv_universal_gemm import (
    HeuristicConfig,
    NVUniversalGemmHeuristics,
)
from torch._inductor.scheduler import Scheduler
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import (
    ceildiv,
    ensure_nv_universal_gemm_available,
    ensure_nvmatmul_heuristics_available,
    run_and_get_code,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
)
from torch.utils._ordered_set import OrderedSet


def _round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def _prep_k(K, scale_size):
    """Prepare K dimension for swizzle requirements (round up ceildiv to multiple of 4)."""
    return _round_up(ceildiv(K, scale_size), 4)


def _create_tensor_with_layout(layout, rows, cols, dtype, device="cuda"):
    """Create a tensor with the specified layout and dtype.

    Supports float16, bfloat16, float8_e4m3fn, and float4_e2m1fn_x2.
    """
    is_fp4 = dtype == torch.float4_e2m1fn_x2
    is_fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    def _make_flat(n):
        if is_fp4:
            return torch.randint(0, 256, (n,), device=device, dtype=torch.uint8).view(
                torch.float4_e2m1fn_x2
            )
        elif is_fp8:
            return torch.randint(-1, 2, (n,), device=device).to(dtype)
        else:
            return torch.randn(n, device=device, dtype=dtype)

    if layout == "contiguous":
        if is_fp4:
            return torch.randint(
                0, 256, (rows, cols), device=device, dtype=torch.uint8
            ).view(torch.float4_e2m1fn_x2)
        elif is_fp8:
            return torch.randint(-1, 2, (rows, cols), device=device).to(dtype)
        else:
            return torch.randn(rows, cols, device=device, dtype=dtype)
    elif layout == "aligned_offset":
        storage = _make_flat(rows * cols + 512)
        offset = 16 // storage.element_size()
        return torch.as_strided(storage[offset:], (rows, cols), (cols, 1))
    elif layout == "view":
        return _make_flat(rows * cols).view(rows, cols)
    elif layout == "padded":
        row_pitch = cols + 8
        storage = _make_flat(rows * row_pitch)
        return torch.as_strided(storage, (rows, cols), (row_pitch, 1))
    else:
        raise ValueError(f"Unknown layout: {layout}")


def _make_nvfp4_scaled_mm_inputs(m, n, k):
    packed_k = k // 2
    a = _create_tensor_with_layout("contiguous", m, packed_k, torch.float4_e2m1fn_x2)
    b = _create_tensor_with_layout("contiguous", n, packed_k, torch.float4_e2m1fn_x2).T
    scale_k = _prep_k(k, 16)
    scale_a = torch.rand(_round_up(m, 128) * scale_k, device="cuda").to(
        torch.float8_e4m3fn
    )
    scale_b = torch.rand(_round_up(n, 128) * scale_k, device="cuda").to(
        torch.float8_e4m3fn
    )
    return a, b, scale_a, scale_b


def _nvgemm_config(**overrides):
    """Standard NVGEMM test config. Always disables ATen fallback."""
    cfg = {
        "max_autotune": True,
        "max_autotune_gemm_backends": "NVGEMM",
        "nvgemm_max_profiling_configs": 3,
        "autotune_fallback_to_aten": False,
    }
    cfg.update(overrides)
    return cfg


# TODO(nikhilap): Remove Blackwell restriction once cutlass_api includes H100 kernels
@unittest.skipIf(
    not (ensure_nv_universal_gemm_available() and is_datacenter_blackwell_arch()),
    "NVIDIA Universal GEMM (cutlass_api) library not available or not on Blackwell",
)
@instantiate_parametrized_tests
class TestNVUniversalGemm(TestCase):
    """Test cases for NVIDIA Universal GEMM functionality."""

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize(
        "layout_a,layout_b",
        (
            ("contiguous", "contiguous"),
            ("aligned_offset", "contiguous"),
            ("contiguous", "aligned_offset"),
            ("contiguous", "view"),
            ("aligned_offset", "view"),
            ("padded", "contiguous"),
            ("contiguous", "padded"),
        ),
    )
    def test_matmul(self, dtype, layout_a, layout_b):
        """Test matmul with various dtypes and tensor layouts.

        M=513 tests that non-divisible M dimension works
        (only N and K must be divisible by 16).
        """
        m, n, k = 513, 512, 512

        def matmul(a, b):
            return a @ b

        a = _create_tensor_with_layout(layout_a, m, k, dtype)
        b = _create_tensor_with_layout(layout_b, k, n, dtype)
        expected = matmul(a, b)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(matmul)
            result = compiled_fn(a, b)

        torch.testing.assert_close(result, expected)

    def test_arch_filter_rejects_min_cc_only_kernels(self):
        """designed_for_min_cc <= device cc is insufficient: an arch-conditional
        sm90 kernel reports min_cc=90 but only lists a cc=90 target and won't
        compile on sm100. The exact-arch filter (via _device_target) must reject
        kernels a min_cc-only check would wrongly accept.
        """
        from torch._inductor.codegen.nv_universal_gemm import kernel_cache

        major, minor = torch.cuda.get_device_capability()
        cc = major * 10 + minor
        device_target = kernel_cache._device_target(cc)

        kernel_cache.clear_cache()
        manifest = kernel_cache._get_kernel_cache()
        min_cc_ok = [k for k in manifest.values() if k.designed_for_min_cc <= cc]
        arch_ok = [
            k
            for k in min_cc_ok
            if device_target.supports_operators_from(k.metadata.supported_targets)
        ]

        self.assertTrue(min_cc_ok, "no kernels pass the min_cc check")
        self.assertLess(
            len(arch_ok),
            len(min_cc_ok),
            "exact-arch filter should reject kernels that min_cc alone accepts",
        )

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_matmul_swap_ab(self, dtype):
        """swap_ab computes (B^T @ A^T)^T so the large N lands on the M-axis,
        improving tile utilization for small-M shapes. Verify a small-M matmul
        stays numerically correct with swap_ab enabled (the swapped operands and
        transposed output view must round-trip to the original result).
        """
        m, n, k = 16, 512, 512

        def matmul(a, b):
            return a @ b

        a = _create_tensor_with_layout("contiguous", m, k, dtype)
        b = _create_tensor_with_layout("contiguous", k, n, dtype)
        expected = matmul(a, b)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config(nvgemm_swap_ab=True)):
            compiled_fn = torch.compile(matmul)
            result = compiled_fn(a, b)

        torch.testing.assert_close(result, expected)

    @parametrize("m", (1, 16))
    def test_matmul_swap_ab_epilogue(self, m):
        n, k = 512, 512

        def matmul(a, b):
            return torch.relu(a @ b)

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        torch._dynamo.reset()
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm import (
            NVUniversalGemmCaller,
        )

        def benchmark(caller, *args, **kwargs):
            if caller.swap_ab and caller.supports_epilogue_fusion:
                return 0.1
            return 0.5 if caller.swap_ab else 1.0

        with (
            config.patch(
                _nvgemm_config(
                    nvgemm_swap_ab=True,
                    benchmark_epilogue_fusion=False,
                )
            ),
            mock.patch.object(
                NVUniversalGemmCaller, "benchmark", autospec=True, side_effect=benchmark
            ),
        ):
            result, (code,) = run_and_get_code(torch.compile(matmul), a, b)

        self.assertEqual(result, matmul(a, b))
        self.assertIn("swap_ab=True", code)
        self.assertIn("EpilogueArguments", code)
        self.assertIn("VendoredDenseGemmEFCOperator", code)

    @parametrize(
        "scale_shape",
        ((16, 512), (1, 512), (16, 1)),
        name_fn=lambda shape: "x".join(map(str, shape)),
    )
    def test_matmul_swap_ab_captured_epilogue(self, scale_shape):
        m, n, k = 16, 512, 512

        def matmul(a, b, scale):
            return (a @ b).float() * scale

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(scale_shape, device="cuda")
        if scale_shape == (16, 1):
            scale = torch.randn(32, 1, device="cuda")[::2]

        torch._dynamo.reset()
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm import (
            NVUniversalGemmCaller,
        )

        def benchmark(caller, *args, **kwargs):
            if caller.swap_ab and caller.supports_epilogue_fusion:
                return 0.1
            return 0.5 if caller.swap_ab else 1.0

        with (
            config.patch(
                _nvgemm_config(
                    nvgemm_swap_ab=True,
                    benchmark_epilogue_fusion=False,
                )
            ),
            mock.patch.object(
                NVUniversalGemmCaller, "benchmark", autospec=True, side_effect=benchmark
            ),
        ):
            result, (code,) = run_and_get_code(torch.compile(matmul), a, b, scale)

        self.assertEqual(result, matmul(a, b, scale), atol=0.7, rtol=1e-2)
        self.assertIn("swap_ab=True", code)
        self.assertIn("EpilogueArguments", code)

    def test_matmul_swap_ab_dynamic_epilogue(self):
        k = 512

        def matmul(a, b):
            return torch.relu((a @ b).float() + 1)

        torch._dynamo.reset()
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm import (
            NVUniversalGemmCaller,
        )

        def benchmark(caller, *args, **kwargs):
            if caller.swap_ab and caller.supports_epilogue_fusion:
                return 0.1
            return 0.5 if caller.swap_ab else 1.0

        with (
            config.patch(
                _nvgemm_config(
                    nvgemm_swap_ab=True,
                    benchmark_epilogue_fusion=False,
                )
            ),
            mock.patch.object(
                NVUniversalGemmCaller, "benchmark", autospec=True, side_effect=benchmark
            ),
        ):
            compiled = torch.compile(matmul, dynamic=True)
            a = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(k, 512, device="cuda", dtype=torch.bfloat16)
            result, (code,) = run_and_get_code(compiled, a, b)
            self.assertEqual(result, matmul(a, b), atol=0.7, rtol=1e-2)

            a = torch.randn(32, k, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(k, 384, device="cuda", dtype=torch.bfloat16)
            self.assertEqual(compiled(a, b), matmul(a, b), atol=0.7, rtol=1e-2)

        self.assertIn("swap_ab=True", code)
        self.assertIn("EpilogueArguments", code)

    def test_cudagraphs_intermediate_addmm(self):
        """An NVGEMM addmm whose bias-epilogue output is an intermediate consumed
        downstream must not leave that output retained by the cached EFC kernel,
        or CUDA graph capture fails ("tensor(s) in the cudagraph pool not tracked
        as outputs") and the tensor leaks. The cached kernel is built from
        meta-device tensors, so runtime outputs flow only through the per-call
        arguments.
        """
        dtype = torch.bfloat16
        m, k = 512, 512
        # Scaled down so the chained bf16 matmul stays well-conditioned.
        bias = torch.randn(k, dtype=dtype, device="cuda") * 0.1
        x = torch.randn(m, k, dtype=dtype, device="cuda") * 0.1
        w = torch.randn(k, k, dtype=dtype, device="cuda") * 0.1

        def chain(bias, x, w):
            # h is an NVGEMM addmm output consumed downstream (not the graph's
            # final output), which is what triggers the retention bug.
            h = torch.addmm(bias, x, w.t())
            return torch.addmm(bias, h, w.t())

        expected = chain(bias.float(), x.float(), w.float()).to(dtype)

        torch._dynamo.reset()

        cfg = _nvgemm_config()
        cfg["triton.cudagraphs"] = True
        with config.patch(cfg):
            compiled_fn = torch.compile(chain)
            for _ in range(3):
                result = compiled_fn(bias, x, w)

        torch.testing.assert_close(result, expected, rtol=1.6e-2, atol=1e-1)

    @unittest.skipIf(IS_FBCODE, "CUTLASS Operator API is not available in fbcode")
    def test_vendored_dense_efc_replacement_has_unique_names(self):
        from cutlass import Float32
        from cutlass.operators.arguments import GemmArguments

        from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
            _args_query_candidates,
        )

        a = torch.empty(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.empty(64, 128, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(128, 128, device="cuda", dtype=torch.bfloat16)
        args = GemmArguments(a, b, out, accumulator_type=Float32)
        major, minor = torch.cuda.get_device_capability()
        kernels = _args_query_candidates(args, major * 10 + minor, efc_only=True)
        names = [kernel.metadata.operator_name for kernel in kernels]

        self.assertTrue(names)
        self.assertEqual(len(names), len(set(names)))

    def test_efc_epilogue_lookup_no_deadlock(self):
        """get_efc_kernel_with_epilogue holds _cache_lock and, when the base EFC
        kernel is not pre-resolved and misses the args cache, calls
        get_kernel_by_name -> _ensure_caches, which re-acquires _cache_lock on the
        same thread. With a plain (non-reentrant) lock this self-deadlocks; the
        lock is an RLock. Run the reentrant path in a worker thread and assert it
        completes (a plain Lock would hang here).
        """
        from torch._inductor.codegen.nv_universal_gemm import kernel_cache

        # Force the manifest-fallback re-entry: no cached manifest, empty args
        # cache, unknown name, no pre-resolved base_kernel.
        kernel_cache.clear_cache()
        result = []

        def worker():
            result.append(
                kernel_cache.get_efc_kernel_with_epilogue(
                    "__nonexistent_kernel__", None, base_kernel=None
                )
            )

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=120)
        self.assertFalse(
            t.is_alive(),
            "get_efc_kernel_with_epilogue deadlocked (reentrant _cache_lock)",
        )
        self.assertIsNone(result[0])

    def test_unaligned_base_pointer_rejected(self):
        """Test that matmul with unaligned base pointer is rejected.

        cutlass_api requires 16-byte aligned base pointers. Since alignment
        can't be checked at compile time (FakeTensors don't have real pointers),
        Inductor must guard against unaligned buffers.
        """
        m, n, k = 512, 512, 512
        dtype = torch.bfloat16
        device = "cuda"

        def matmul(a, b):
            return a @ b

        # offset=117 elements * 2 bytes = 234 bytes (NOT 16-byte aligned)
        storage = torch.randn(m * k + 512, device=device, dtype=dtype)
        a = torch.as_strided(storage[117:], (m, k), (k, 1))
        b = torch.randn(k, n, device=device, dtype=dtype)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(matmul)
            with self.assertRaisesRegex(
                Exception, "NoValidChoicesError|no valid choice"
            ):
                compiled_fn(a, b)

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_reinterpret_view_from_slice(self, dtype):
        """Test that sliced tensors (creating ReinterpretViews) work correctly.

        When tensors are slices of a shared buffer (e.g., from a fused projection),
        they become ReinterpretViews with non-contiguous strides.
        """
        m, n, k = 512, 512, 512
        device = "cuda"

        def fn(x, weight):
            projected = x @ weight  # (m, 2*n)
            a, b = projected.split(n, dim=1)  # Each is (m, n)
            return a @ b.t()  # (m, m)

        x = torch.randn(m, k, device=device, dtype=dtype)
        weight = torch.randn(k, 2 * n, device=device, dtype=dtype)
        expected = fn(x, weight)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(fn)
            result = compiled_fn(x, weight)

        torch.testing.assert_close(result, expected)

    def test_workspace_allocation(self):
        """Test that workspace allocation works correctly.

        Since no current CUTLASS kernels require a workspace, we mock the
        kernel.get_workspace_size method to return a non-zero value.
        """
        m, n, k = 512, 512, 512
        dtype = torch.bfloat16
        device = "cuda"

        def matmul(a, b):
            return a @ b

        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)
        expected = matmul(a, b)

        torch._dynamo.reset()

        import cutlass_api

        def patched_get_workspace_size(self, args):
            return 1024

        with patch.object(
            cutlass_api.Kernel,
            "get_workspace_size",
            patched_get_workspace_size,
        ):
            with config.patch(_nvgemm_config()):
                result, (code,) = run_and_get_code(
                    torch.compile(matmul),
                    a,
                    b,
                )

        self.assertIn("workspace=workspace", code)
        torch.testing.assert_close(result, expected)

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_bmm_non_standard_batch_stride(self, dtype):
        """Test BMM path with non-standard batch strides."""
        batch, m, n, k = 8, 64, 256, 128
        device = "cuda"

        def bmm(a, b):
            return torch.bmm(a, b)

        # Transpose creates non-largest batch stride
        a_base = torch.randn(m, batch, k, device=device, dtype=dtype)
        a = a_base.transpose(0, 1)  # (batch, m, k) with stride (k, batch*k, 1)

        b_base = torch.randn(k, batch, n, device=device, dtype=dtype)
        b = b_base.transpose(0, 1)  # (batch, k, n) with stride (n, batch*n, 1)

        if a.stride()[0] == a.shape[1] * a.shape[2]:
            raise AssertionError(
                "Test setup error: a should have non-standard batch stride"
            )
        if b.stride()[0] == b.shape[1] * b.shape[2]:
            raise AssertionError(
                "Test setup error: b should have non-standard batch stride"
            )

        expected = bmm(a, b)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(bmm)
            result = compiled_fn(a, b)

        torch.testing.assert_close(result, expected)

    @parametrize(
        "layout_a",
        ("contiguous", "aligned_offset", "view"),
    )
    @parametrize(
        "m,n,k",
        (
            (256, 512, 1024),
            (256, 1024, 512),
            (128, 256, 512),
            (512, 256, 1024),
        ),
    )
    def test_scaled_gemm_mxfp8(self, layout_a, m, n, k):
        """Test MXFP8 scaled GEMM with NVGEMM backend."""
        block_size = 32

        def scaled_mm(a, b, scale_a, scale_b):
            return torch._scaled_mm(
                a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32
            )

        a_fp8 = _create_tensor_with_layout(layout_a, m, k, torch.float8_e4m3fn)
        b_fp8 = torch.randint(-1, 2, (n, k), device="cuda").to(torch.float8_e4m3fn).T

        scale_a = torch.rand(m, _prep_k(k, block_size), device="cuda").to(
            torch.float8_e8m0fnu
        )
        scale_b = torch.rand(_prep_k(k, block_size), n, device="cuda").to(
            torch.float8_e8m0fnu
        )

        expected = scaled_mm(a_fp8, b_fp8, scale_a, scale_b)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(scaled_mm)
            result = compiled_fn(a_fp8, b_fp8, scale_a, scale_b)

        torch.testing.assert_close(result, expected)

    @parametrize(
        "m,n,k",
        (
            (16, 512, 1024),
            (32, 256, 512),
            (64, 512, 1024),
            (64, 64, 512),
        ),
    )
    def test_scaled_gemm_nvf4_padded_scales(self, m, n, k):
        """Test NVF4 with padded scales (M or N < 128).

        torch._scaled_mm creates scales with block_size_mn=128 padding,
        producing more elements than the kernel expects. The kernel's
        _supports() must accept both padded and unpadded scale numels.
        """
        packed_k = k // 2

        def scaled_mm(a, b, scale_a, scale_b):
            return torch._scaled_mm(
                a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
            )

        a_fp4 = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b_fp4 = torch.randint(
            0, 256, (n, packed_k), device="cuda", dtype=torch.uint8
        ).view(torch.float4_e2m1fn_x2)
        b_fp4_t = b_fp4.T

        num_k_blocks = ceildiv(k, 16)
        padded_k_blocks = _round_up(num_k_blocks, 4)
        scale_a_numel = _round_up(m, 128) * padded_k_blocks
        scale_b_numel = _round_up(n, 128) * padded_k_blocks

        scale_a = torch.rand(scale_a_numel, device="cuda").to(torch.float8_e4m3fn)
        scale_b = torch.rand(scale_b_numel, device="cuda").to(torch.float8_e4m3fn)

        expected = scaled_mm(a_fp4, b_fp4_t, scale_a, scale_b)

        torch._dynamo.reset()

        with config.patch(
            _nvgemm_config(
                **{"test_configs.autotune_choice_desc_regex": "inductor_vendored"}
            )
        ):
            compiled_fn = torch.compile(scaled_mm)
            result = compiled_fn(a_fp4, b_fp4_t, scale_a, scale_b)

        torch.testing.assert_close(result, expected, equal_nan=True)

    def test_scaled_gemm_grouped_n_reduce_provider(self):
        from cutlass import Float32
        from cutlass.operators import ScaleMode, ScaleSwizzleMode

        from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
            _scaled_candidates,
        )
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            _create_gemm_arguments,
        )
        from torch._inductor.kernel.gemm_epilogue import GemmReductionArguments

        m, n, k = 128, 128, 512
        packed_k = k // 2
        a = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b = torch.randint(0, 256, (n, packed_k), device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        b = b.T
        padded_k_blocks = _round_up(ceildiv(k, 16), 4)
        scale_a = torch.rand(_round_up(m, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        scale_b = torch.rand(_round_up(n, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        group = 32
        reduce_out = torch.empty((m, n // group), device="cuda", dtype=torch.float32)
        mode = ScaleMode.Blockwise1x16
        swizzle = ScaleSwizzleMode.Swizzle32x4x4
        args = _create_gemm_arguments(
            "SCALED_GEMM",
            (a, b, scale_a, scale_b),
            out,
            Float32,
            scale_mode_a=mode,
            swizzle_mode_a=swizzle,
            scale_mode_b=mode,
            swizzle_mode_b=swizzle,
            local_reduce=GemmReductionArguments(
                output=reduce_out,
                group=group,
                axis=1,
            ),
        )
        kernel = next(
            candidate
            for candidate in _scaled_candidates(args, 100, efc_only=True)
            if candidate.metadata.design.tile_shape[1] == n
        )
        artifact = kernel.compile(args)
        kernel.run(
            args,
            artifact,
            stream=torch.cuda.current_stream(),
            assume_supported_args=True,
        )
        torch.cuda.synchronize()
        self.assertEqual(
            reduce_out,
            out.float().view(m, -1, group).sum(-1),
        )

    def test_scaled_gemm_unit_dim_epilogue_non_current_stream(self):
        from cutlass import Float32
        from cutlass.operators import ScaleMode, ScaleSwizzleMode
        from cutlass.operators.arguments import EpilogueArguments

        from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
            _scaled_candidates,
        )
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            _create_gemm_arguments,
        )

        m, n, k = 1, 128, 512
        packed_k = k // 2
        a = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b = torch.randint(0, 256, (n, packed_k), device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        b = b.T
        padded_k_blocks = _round_up(ceildiv(k, 16), 4)
        scale_a = torch.rand(_round_up(m, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        scale_b = torch.rand(_round_up(n, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        bias = torch.randn((1, 1), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        expected = (
            torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            + bias
        )
        epilogue = EpilogueArguments(
            epilogue_fn="def epilogue(accum, bias):\n    D = accum + bias\n    return D\n",
            bias=bias,
            D=out,
        )
        mode = ScaleMode.Blockwise1x16
        swizzle = ScaleSwizzleMode.Swizzle32x4x4
        args = _create_gemm_arguments(
            "SCALED_GEMM",
            (a, b, scale_a, scale_b),
            out,
            Float32,
            scale_mode_a=mode,
            swizzle_mode_a=swizzle,
            scale_mode_b=mode,
            swizzle_mode_b=swizzle,
            epilogue=epilogue,
        )
        selection_args = _create_gemm_arguments(
            "SCALED_GEMM",
            (a, b, scale_a, scale_b),
            out,
            Float32,
            scale_mode_a=mode,
            swizzle_mode_a=swizzle,
            scale_mode_b=mode,
            swizzle_mode_b=swizzle,
        )
        kernel = next(
            candidate
            for candidate in _scaled_candidates(selection_args, 100, efc_only=True)
            if candidate.metadata.design.tile_shape[1] == n
        )
        artifact = kernel.compile(args)
        torch.cuda.synchronize()
        torch.cuda._sleep(10_000_000)
        stream = torch.cuda.Stream()
        kernel.run(args, artifact, stream=stream, assume_supported_args=True)
        stream.synchronize()
        self.assertEqual(out, expected)

    @parametrize("out_dtype", (torch.float32, torch.bfloat16))
    @parametrize(
        "layout_a",
        ("contiguous", "aligned_offset", "view"),
    )
    @parametrize(
        "m,n,k",
        (
            (256, 512, 1024),
            (256, 1024, 512),
            (128, 256, 512),
            (512, 256, 1024),
        ),
    )
    def test_scaled_gemm_nvf4(self, out_dtype, layout_a, m, n, k):
        """Test NVF4 (Float4 + Float8E4M3FN scales, block_size=16) with NVGEMM backend."""
        packed_k = k // 2
        block_size = 16

        def scaled_mm(a, b, scale_a, scale_b):
            return torch._scaled_mm(
                a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype
            )

        a_fp4 = _create_tensor_with_layout(
            layout_a, m, packed_k, torch.float4_e2m1fn_x2
        )
        b_fp4 = torch.randint(
            0, 256, (n, packed_k), device="cuda", dtype=torch.uint8
        ).view(torch.float4_e2m1fn_x2)
        b_fp4_t = b_fp4.T

        num_k_blocks = ceildiv(k, block_size)
        padded_k_blocks = _round_up(num_k_blocks, 4)
        block_size_mn = 128
        scale_a_numel = block_size_mn * ceildiv(m, block_size_mn) * padded_k_blocks
        scale_b_numel = block_size_mn * ceildiv(n, block_size_mn) * padded_k_blocks

        scale_a = torch.rand(scale_a_numel, device="cuda").to(torch.float8_e4m3fn)
        scale_b = torch.rand(scale_b_numel, device="cuda").to(torch.float8_e4m3fn)

        expected = scaled_mm(a_fp4, b_fp4_t, scale_a, scale_b)

        torch._dynamo.reset()

        with config.patch(
            _nvgemm_config(
                **{"test_configs.autotune_choice_desc_regex": "inductor_vendored"}
            )
        ):
            compiled_fn = torch.compile(scaled_mm)
            result = compiled_fn(a_fp4, b_fp4_t, scale_a, scale_b)

        # a_fp4 and b_fp4_t could come with NaNs.
        torch.testing.assert_close(result, expected, equal_nan=True)

    @parametrize(
        "layout_a",
        ("contiguous", "aligned_offset", "view", "padded"),
    )
    def test_grouped_gemm(self, layout_a):
        """Test grouped GEMM with NVGEMM backend and various A layouts.

        GroupedGemm currently only supports TN layout (column-major B).
        """
        g, k, n = 4, 256, 256
        dtype = torch.bfloat16
        device = "cuda"

        def grouped_mm(a, b, offsets):
            return torch._grouped_mm(a, b, offs=offsets)

        b = torch.randn(g, n, k, device=device, dtype=dtype).permute(0, 2, 1)

        m_per_group = [64, 64, 64, 64]
        total_m = sum(m_per_group)
        offsets = torch.tensor(
            [sum(m_per_group[: i + 1]) for i in range(g)],
            device=device,
            dtype=torch.int32,
        )
        a = _create_tensor_with_layout(layout_a, total_m, k, dtype, device)

        expected = grouped_mm(a, b, offsets)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(grouped_mm)
            result = compiled_fn(a, b, offsets)

        torch.testing.assert_close(result, expected)

    def test_grouped_gemm_varying_offsets(self):
        """Test that different offset distributions produce correct results.

        Runs the same compiled function with two different offset distributions
        (same total_m) to verify offsets are handled dynamically at runtime.
        """
        g, k, n = 4, 256, 256
        dtype = torch.bfloat16
        device = "cuda"

        def grouped_mm(a, b, offsets):
            return torch._grouped_mm(a, b, offs=offsets)

        b = torch.randn(g, n, k, device=device, dtype=dtype).permute(0, 2, 1)

        m_per_group_1 = [64, 64, 64, 64]
        total_m = sum(m_per_group_1)
        offsets_1 = torch.tensor(
            [sum(m_per_group_1[: i + 1]) for i in range(g)],
            device=device,
            dtype=torch.int32,
        )
        a_1 = torch.randn(total_m, k, device=device, dtype=dtype)

        m_per_group_2 = [32, 96, 48, 80]
        if sum(m_per_group_2) != total_m:
            raise AssertionError("Total M must match for cache key test")
        offsets_2 = torch.tensor(
            [sum(m_per_group_2[: i + 1]) for i in range(g)],
            device=device,
            dtype=torch.int32,
        )
        a_2 = torch.randn(total_m, k, device=device, dtype=dtype)

        expected_1 = grouped_mm(a_1, b, offsets_1)
        expected_2 = grouped_mm(a_2, b, offsets_2)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(grouped_mm)

            result_1 = compiled_fn(a_1, b, offsets_1)
            torch.testing.assert_close(result_1, expected_1)

            result_2 = compiled_fn(a_2, b, offsets_2)
            torch.testing.assert_close(result_2, expected_2)

            self.assertFalse(torch.allclose(result_1, result_2))


@instantiate_parametrized_tests
class TestNVUniversalGemmHeuristics(TestCase):
    """Unit tests for NVUniversalGemmHeuristics without requiring actual libraries."""

    @parametrize("tuned", (False, True))
    def test_flex_gemm_nvgemm_tuned_config(self, tuned):
        from torch._inductor.kernel.flex_gemm import lowering

        graph_module = MagicMock()
        subgraph = MagicMock(graph_module=graph_module)

        def observe_config(_graph_module, _args):
            return {
                "max_autotune": config.max_autotune,
                "backends": config.max_autotune_gemm_backends,
                "max_configs": config.nvgemm_max_profiling_configs,
                "supplements": config.nvgemm_supplement_configs,
                "swap_ab": config.nvgemm_swap_ab,
            }

        outer_config = {
            "nvgemm_max_profiling_configs": 3,
            "nvgemm_supplement_configs": False,
            "nvgemm_swap_ab": False,
        }
        with (
            config.patch(outer_config),
            mock.patch.object(lowering, "decompose_nvgemm_additive_gemm"),
            mock.patch.object(
                lowering, "process_subgraph_nodes", side_effect=observe_config
            ),
        ):
            observed = lowering.flex_gemm_lowering.__wrapped__(
                None,
                subgraph,
                (),
                {},
                {"backend": "NVGEMM", "tuned": tuned},
            )
            self.assertEqual(observed["max_autotune"], True)
            self.assertEqual(observed["backends"], "NVGEMM")
            self.assertEqual(observed["max_configs"], None if tuned else 3)
            self.assertEqual(observed["supplements"], tuned)
            self.assertEqual(observed["swap_ab"], tuned)
            self.assertEqual(config.nvgemm_max_profiling_configs, 3)
            self.assertEqual(config.nvgemm_supplement_configs, False)
            self.assertEqual(config.nvgemm_swap_ab, False)

    def test_dense_reduction_capabilities(self):
        import dataclasses

        from torch._inductor.codegen.nv_universal_gemm.epilogue_capabilities import (
            DENSE_GEMM_REDUCTION_CAPABILITIES,
        )
        from torch._inductor.kernel.gemm_epilogue import GemmReductionPlan

        capabilities = DENSE_GEMM_REDUCTION_CAPABILITIES
        self.assertTrue(capabilities.supports("variance_affine:1.0:0.0", "square"))
        self.assertFalse(capabilities.supports("variance_affine:1.0", "square"))
        self.assertTrue(capabilities.supports("max", "abs_scale"))
        self.assertFalse(capabilities.supports("normalize_sum_affine", "identity"))
        self.assertTrue(
            capabilities.supports(
                "normalize_sum_affine:1:0:1:0", "identity", feeds_main=True
            )
        )
        self.assertFalse(
            capabilities.supports(
                "normalize_sum_affine:1:2:3", "identity", feeds_main=True
            )
        )
        self.assertFalse(capabilities.supports("sum", "unsupported"))
        self.assertFalse(capabilities.supports("unsupported", "identity"))

        from torch._inductor.codegen.nv_universal_gemm import GemmVariant

        plan = GemmReductionPlan(
            reduction_output=None,
            group=4,
            axis=1,
            reduction_type="sum",
            source_type="identity",
            primary_output="out",
        )
        self.assertTrue(GemmVariant.GEMM.supports_reduction(plan))
        non_power_of_two = dataclasses.replace(plan, group=3)
        self.assertFalse(GemmVariant.GEMM.supports_reduction(non_power_of_two))
        self.assertFalse(GemmVariant.SCALED_GEMM.supports_reduction(non_power_of_two))
        wide_consumer = dataclasses.replace(
            plan,
            group=64,
            feeds_main=True,
            feed_output="feed",
            consumer_fn="generated_consumer",
        )
        self.assertFalse(GemmVariant.GEMM.supports_reduction(wide_consumer))
        self.assertTrue(GemmVariant.SCALED_GEMM.supports_reduction(wide_consumer))
        unsupported = dataclasses.replace(plan, reduction_type="unsupported")
        self.assertFalse(GemmVariant.GEMM.supports_reduction(unsupported))
        self.assertTrue(GemmVariant.SCALED_GEMM.supports_reduction(plan))
        direct_bool = dataclasses.replace(plan, reduction_type="direct_bool_gt_zero")
        self.assertFalse(GemmVariant.SCALED_GEMM.supports_reduction(direct_bool))
        logsumexp = dataclasses.replace(plan, reduction_type="logsumexp")
        self.assertFalse(GemmVariant.SCALED_GEMM.supports_reduction(logsumexp))
        variance = dataclasses.replace(plan, reduction_type="variance_affine:1:0")
        self.assertFalse(GemmVariant.SCALED_GEMM.supports_reduction(variance))
        online_softmax = dataclasses.replace(plan, reduction_type="online_softmax")
        self.assertFalse(GemmVariant.SCALED_GEMM.supports_reduction(online_softmax))
        self.assertFalse(
            GemmVariant.SCALED_GEMM.supports_reduction(
                dataclasses.replace(online_softmax, feeds_main=True)
            )
        )
        secondary = dataclasses.replace(plan, secondary_feed_output="secondary")
        self.assertFalse(GemmVariant.GEMM.supports_reduction(secondary))
        self.assertFalse(GemmVariant.SCALED_GEMM.supports_reduction(secondary))
        direct_secondary = dataclasses.replace(
            secondary, secondary_feed_type="direct_bool_gt_zero"
        )
        self.assertTrue(GemmVariant.GEMM.supports_reduction(direct_secondary))
        custom_secondary = dataclasses.replace(
            secondary, secondary_consumer_fn="generated_consumer"
        )
        self.assertTrue(GemmVariant.GEMM.supports_reduction(custom_secondary))
        self.assertFalse(GemmVariant.SCALED_GEMM.supports_reduction(custom_secondary))
        normalized_secondary = dataclasses.replace(
            secondary,
            axis=0,
            feeds_main=True,
            secondary_feed_type="normalize_sum_affine:1:0:1:0",
        )
        self.assertTrue(GemmVariant.GEMM.supports_reduction(normalized_secondary))
        self.assertFalse(
            GemmVariant.GEMM.supports_reduction(
                dataclasses.replace(normalized_secondary, axis=1)
            )
        )
        self.assertFalse(GemmVariant.SCALED_GEMM.supports_reduction(unsupported))
        self.assertFalse(GemmVariant.GROUPED_GEMM.supports_reduction(plan))

    def test_reduction_epilogue_runs_full_fusion_validation(self):
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_scheduling import (
            NVUniversalGemmScheduling,
        )
        from torch._inductor.ir import Buffer

        scheduling = object.__new__(NVUniversalGemmScheduling)
        template = mock.Mock(spec=Buffer)
        template_snode = mock.Mock()
        node1 = mock.Mock()
        node1.get_template_node.return_value = template
        node1.get_nodes.return_value = (template_snode,)
        node2 = mock.Mock()
        node2.get_nodes.return_value = (mock.Mock(),)
        program = mock.Mock(supported=True, feeds_main=False)
        program.reduction_partition.intersects.return_value = True

        with (
            mock.patch.object(scheduling, "_lower_epilogue", return_value=program),
            mock.patch.object(
                scheduling,
                "is_nv_universal_gemm_template",
                side_effect=lambda node: node is template_snode,
            ),
            mock.patch.object(
                scheduling, "_can_fuse_epilogue_impl", return_value=False
            ) as validate,
        ):
            self.assertFalse(scheduling.can_fuse_reduction_epilogue(node1, node2))

        validate.assert_called_once_with(template_snode, [], node2)

    def test_bool_output_only_blocks_nvgemm_horizontal_fusion(self):
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_scheduling import (
            NVUniversalGemmScheduling,
        )

        scheduling = object.__new__(NVUniversalGemmScheduling)
        node = mock.Mock()
        node.read_writes.reads = ()

        with (
            mock.patch.object(scheduling, "has_bool_output", return_value=True),
            mock.patch.object(
                scheduling, "is_nv_universal_gemm_template", return_value=False
            ),
            mock.patch.object(
                scheduling, "is_nv_universal_gemm_fused_template", return_value=False
            ),
        ):
            self.assertFalse(scheduling.has_nvgemm_bool_output(node))

        with (
            mock.patch.object(scheduling, "has_bool_output", return_value=True),
            mock.patch.object(
                scheduling, "is_nv_universal_gemm_template", return_value=True
            ),
        ):
            self.assertTrue(scheduling.has_nvgemm_bool_output(node))

    def test_epilogue_program_derived_properties(self):
        import dataclasses

        from torch._inductor.codegen.nv_universal_gemm.epilogue_lowering import (
            NVGemmEpilogueCapture,
            NVGemmEpilogueProgram,
            NVGemmReductionPartition,
            NVGemmReductionRegion,
        )
        from torch._inductor.kernel.gemm_epilogue import (
            GemmReductionConfig,
            GemmReductionPlan,
        )

        config = GemmReductionConfig(
            output_name="reduction",
            group=4,
            axis=1,
            reduction_type="sum",
            source_type="identity",
        )
        region = NVGemmReductionRegion(config=config, nodes=())
        partition = NVGemmReductionPartition(regions=(region,))
        plan = GemmReductionPlan(
            reduction_output=None,
            group=8,
            axis=1,
            reduction_type="sum",
            source_type="identity",
            primary_output="out",
            feeds_main=True,
        )
        capture = NVGemmEpilogueCapture(gemm=mock.Mock(), nodes=(), analysis=None)
        child = mock.Mock(node=object())
        fused = mock.Mock()
        fused.get_nodes.return_value = (child,)
        normalized = NVGemmEpilogueCapture.from_nodes(mock.Mock(), (fused,))
        self.assertEqual(normalized.nodes, (child,))
        self.assertIsNone(normalized.analysis)
        program = NVGemmEpilogueProgram(
            capture=capture, reduction_partition=partition, reduction_plan=plan
        )
        self.assertEqual(program.min_tile_n, 8)
        self.assertEqual(
            dataclasses.replace(program, reduction_plan=None).min_tile_n,
            4,
        )
        self.assertEqual(
            dataclasses.replace(
                program, reduction_plan=dataclasses.replace(plan, feeds_main=False)
            ).min_tile_n,
            4,
        )

        source = mock.Mock()
        finalizer = mock.Mock()
        unrelated = mock.Mock()
        partition = NVGemmReductionPartition(
            regions=(
                NVGemmReductionRegion(
                    config=config,
                    nodes=(source, finalizer),
                ),
            )
        )
        self.assertTrue(partition.intersects((source,)))
        self.assertTrue(partition.intersects((finalizer,)))
        self.assertFalse(partition.intersects((unrelated,)))
        self.assertTrue(partition.owns((source, finalizer)))
        self.assertFalse(partition.owns((source, unrelated)))
        self.assertIs(partition.region_for((source,)), partition.regions[0])
        self.assertIs(partition.region_for((source, finalizer)), partition.regions[0])
        self.assertIsNone(partition.region_for((finalizer,)))
        self.assertIsNone(partition.region_for((unrelated,)))

    def test_reduction_contracts_validate_geometry(self):
        from torch._inductor.kernel.gemm_epilogue import (
            GemmReductionConfig,
            GemmReductionPlan,
        )

        with self.assertRaisesRegex(RuntimeError, "group must be positive"):
            GemmReductionConfig(
                output_name="reduction",
                group=0,
                axis=1,
                reduction_type="sum",
                source_type="identity",
            )
        with self.assertRaisesRegex(RuntimeError, "axis must be 0 or 1"):
            GemmReductionPlan(
                reduction_output="reduction",
                group=4,
                axis=2,
                reduction_type="sum",
                source_type="identity",
                primary_output="out",
            )

    def test_epilogue_program_owns_only_named_feed_intermediates(self):
        from torch._inductor.codegen.nv_universal_gemm import epilogue_lowering
        from torch._inductor.codegen.nv_universal_gemm.epilogue_lowering import (
            NVGemmEpilogueCapture,
            NVGemmEpilogueProgram,
            NVGemmReductionPartition,
        )
        from torch._inductor.kernel.gemm_epilogue import GemmReductionPlan

        class FakeBuffer:
            def __init__(self, name):
                self.name = name
                self.data = FakeMultiOutputReduction()

            def get_name(self):
                return self.name

        class FakeMultiOutputReduction:
            pass

        feed = mock.Mock(node=FakeBuffer("softmax_intermediate"))
        unrelated = mock.Mock(node=FakeBuffer("unrelated_reduction"))
        capture = NVGemmEpilogueCapture(
            gemm=mock.Mock(), nodes=(feed, unrelated), analysis=None
        )
        plan = GemmReductionPlan(
            reduction_output=None,
            group=4,
            axis=1,
            reduction_type="online_softmax",
            source_type="identity",
            primary_output="out",
            feeds_main=True,
        )
        program = NVGemmEpilogueProgram(
            capture=capture,
            reduction_partition=NVGemmReductionPartition(()),
            reduction_plan=plan,
            intermediate_outputs=("softmax_intermediate",),
        )
        with (
            mock.patch.object(epilogue_lowering, "Buffer", FakeBuffer),
            mock.patch.object(epilogue_lowering, "ComputedBuffer", FakeBuffer),
            mock.patch.object(
                epilogue_lowering, "MultiOutputReduction", FakeMultiOutputReduction
            ),
        ):
            self.assertEqual(program.owned_nodes, (feed,))
            self.assertEqual(program.pointwise_nodes, (unrelated,))

    def test_static_reduction_plan_composition(self):
        import dataclasses

        from torch._inductor.codegen.nv_universal_gemm.epilogue_lowering import (
            NVGemmEpilogueLowering,
            NVGemmFeedPlan,
            NVGemmReductionPartition,
            NVGemmReductionRegion,
        )
        from torch._inductor.kernel.gemm_epilogue import (
            GemmReductionConfig,
            GemmReductionPlan,
        )

        context = mock.Mock()
        context.gemm.get_name.return_value = "gemm"
        config = GemmReductionConfig(
            output_name="reduced",
            group=4,
            axis=1,
            reduction_type="sum",
            source_type="identity",
        )
        partition = NVGemmReductionPartition(
            (NVGemmReductionRegion(config=config, nodes=()),)
        )
        plan = GemmReductionPlan(
            reduction_output=None,
            group=4,
            axis=1,
            reduction_type="sum",
            source_type="identity",
            primary_output="out",
            feeds_main=True,
        )

        combined = NVGemmEpilogueLowering._static_reduction_plan(
            context, partition, NVGemmFeedPlan(plan=plan)
        )
        self.assertEqual(
            combined, dataclasses.replace(plan, reduction_output="reduced")
        )
        incompatible = dataclasses.replace(plan, group=8)
        self.assertIsNone(
            NVGemmEpilogueLowering._static_reduction_plan(
                context, partition, NVGemmFeedPlan(plan=incompatible)
            )
        )

        bool_config = dataclasses.replace(
            config,
            output_name="mask",
            reduction_type="direct_bool_gt_zero",
        )
        bool_partition = NVGemmReductionPartition(
            (NVGemmReductionRegion(config=bool_config, nodes=()),)
        )
        combined = NVGemmEpilogueLowering._static_reduction_plan(
            context, bool_partition, NVGemmFeedPlan(plan=plan)
        )
        self.assertEqual(combined.secondary_feed_output, "mask")
        self.assertEqual(combined.secondary_feed_type, "direct_bool_gt_zero")

    def test_grouped_reduction_conversion_contract(self):
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRExpression as Expr,
            GemmEpilogueIRStore,
            grouped_reduction_ir,
        )

        load = Expr("load", ("gemm", 0, None))

        def classify(value):
            square = Expr("mul", (value, value))
            reduction = Expr("reduction", (torch.float32, torch.float32, "sum", square))
            return grouped_reduction_ir(
                GemmEpilogueIRStore(0, reduction), "gemm", 4, torch.bfloat16
            )

        fp32 = Expr("to_dtype", (load, torch.float32))
        self.assertEqual(classify(fp32), ("sum", "square"))
        fp16_then_fp32 = Expr(
            "to_dtype", (Expr("to_dtype", (load, torch.float16)), torch.float32)
        )
        self.assertIsNone(classify(fp16_then_fp32))
        bitcast = Expr("to_dtype_bitcast", (load, torch.bfloat16, torch.bfloat16))
        self.assertIsNone(classify(Expr("to_dtype", (bitcast, torch.float32))))

        def classify_unrolled(value):
            values = [Expr("load", ("gemm", offset, None)) for offset in range(4)]
            values = [value(item) for item in values]
            result = values[0]
            for item in values[1:]:
                result = Expr("add", (result, item))
            return grouped_reduction_ir(
                GemmEpilogueIRStore(0, result), "gemm", 4, torch.bfloat16
            )

        self.assertIsNone(
            classify_unrolled(
                lambda value: Expr(
                    "to_dtype",
                    (Expr("to_dtype", (value, torch.float16)), torch.float32),
                )
            )
        )
        self.assertIsNone(
            classify_unrolled(
                lambda value: Expr(
                    "to_dtype_bitcast", (value, torch.bfloat16, torch.bfloat16)
                )
            )
        )

    def test_grouped_reduction_ir_normalizes_loop_representation(self):
        from torch._inductor.kernel.gemm_epilogue import GemmReductionConfig
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRAnalysis,
            GemmEpilogueIRExpression as Expr,
            GemmEpilogueIRStore,
        )

        load = Expr("load", ("gemm", 0, None))
        reduction = Expr("reduction", (torch.float32, torch.float32, "sum", load))
        analysis = GemmEpilogueIRAnalysis({"out": GemmEpilogueIRStore(0, reduction)})

        self.assertEqual(
            analysis.grouped_reduction("out", "gemm", 4, 1, torch.float32),
            GemmReductionConfig(
                output_name="out",
                group=4,
                axis=1,
                reduction_type="sum",
                source_type="identity",
            ),
        )

    def test_grouped_reduction_rejects_ambiguous_composite(self):
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRExpression as Expr,
            GemmEpilogueIRStore,
            grouped_reduction_ir,
        )

        summed = Expr(
            "reduction",
            (torch.float32, torch.float32, "sum", Expr("load", ("gemm", 0, None))),
        )
        maximum = Expr(
            "reduction",
            (torch.float32, torch.float32, "max", Expr("load", ("gemm", 0, None))),
        )
        store = GemmEpilogueIRStore(0, Expr("add", (summed, maximum)))
        self.assertIsNone(grouped_reduction_ir(store, "gemm", 4, torch.float32))

    def test_epilogue_ir_preserves_empty_tuple_argument(self):
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRExpression,
        )

        expression = GemmEpilogueIRExpression("reshape", ("value", ()))
        self.assertEqual(expression.args, ("value", ()))
        self.assertEqual(expression.kwargs, ())

    def test_local_reduce_cache_specialization(self):
        import dataclasses

        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            _local_reduce_specialization,
        )
        from torch._inductor.kernel.gemm_epilogue import (
            GemmReductionArguments,
            GemmReductionPlan,
        )

        argument_fields = {
            field.name for field in dataclasses.fields(GemmReductionArguments)
        }
        classified_fields = set(GemmReductionArguments.TENSOR_FIELDS) | set(
            GemmReductionArguments.SPECIALIZATION_FIELDS
        )
        self.assertEqual(argument_fields, classified_fields)
        plan_fields = {field.name for field in dataclasses.fields(GemmReductionPlan)}
        self.assertTrue(
            set(GemmReductionArguments.SPECIALIZATION_FIELDS).issubset(plan_fields)
        )

        base = GemmReductionArguments(group=4, reduction_type="mean")
        mapped = dataclasses.replace(
            base, output="output", secondary_feed_output="secondary"
        ).map_tensors(str.upper)
        self.assertEqual(mapped.output, "OUTPUT")
        self.assertIsNone(mapped.feed_output)
        self.assertEqual(mapped.secondary_feed_output, "SECONDARY")
        self.assertEqual(mapped.group, base.group)

        plan = GemmReductionPlan(
            reduction_output=None,
            group=8,
            axis=0,
            reduction_type="max",
            source_type="abs",
            primary_output="out",
            feeds_main=True,
        )
        from_plan = GemmReductionArguments.from_plan(plan, output="output")
        self.assertEqual(from_plan.output, "output")
        self.assertEqual(from_plan.group, plan.group)
        self.assertEqual(from_plan.reduction_type, plan.reduction_type)

        specialization = _local_reduce_specialization({"local_reduce": base})
        for field, value in (
            ("group", 8),
            ("axis", 0),
            ("feeds_main", True),
            ("secondary_feed_type", "direct_bool_gt_zero"),
        ):
            variant = dataclasses.replace(base, **{field: value})
            self.assertNotEqual(
                specialization,
                _local_reduce_specialization({"local_reduce": variant}),
            )

        tensor = torch.empty((4, 8))
        tensor_specialization = _local_reduce_specialization(
            {"local_reduce": dataclasses.replace(base, output=tensor)}
        )
        self.assertNotEqual(
            tensor_specialization,
            _local_reduce_specialization({"local_reduce": base}),
        )
        self.assertNotEqual(
            tensor_specialization,
            _local_reduce_specialization(
                {"local_reduce": dataclasses.replace(base, output=torch.empty((8, 4)))}
            ),
        )
        self.assertNotEqual(
            tensor_specialization,
            _local_reduce_specialization(
                {"local_reduce": dataclasses.replace(base, feed_output=tensor)}
            ),
        )

    def test_reduction_compile_config_preserves_callback_group(self):
        from torch._inductor.kernel import gemm_epilogue_codegen

        reduction = mock.Mock(
            reduce_op="reduce",
            init_val="init",
            combine="combine",
            source="source",
            finalize="finalize",
        )
        args = mock.Mock(
            group=4,
            axis=1,
            reduction_type="sum",
            source_type="identity",
            feeds_main=True,
            secondary_feed_type="direct_bool_gt_zero",
        )
        config = gemm_epilogue_codegen.GemmReductionCompileConfig(
            args=args,
            reduction=reduction,
            consumer="consumer",
            secondary_consumer="secondary_consumer",
        )

        self.assertIs(config.args, args)
        common = (4, 1, "sum", "identity", True)
        primary = ("reduce", "init", "combine", "source", "finalize", "consumer")
        self.assertEqual(
            config.constexprs(),
            (*common, "direct_bool_gt_zero", *primary, "secondary_consumer"),
        )

    def test_local_reduce_plan_deduplicates_outputs(self):
        from torch._inductor.kernel.gemm_epilogue import (
            GemmReductionDescriptor,
            GemmReductionPlan,
        )

        plan = GemmReductionPlan(
            reduction_output="aux",
            group=4,
            axis=1,
            reduction_type="sum",
            source_type="identity",
            primary_output="output",
            feed_output="aux",
            secondary_feed_output="output",
        )
        self.assertEqual(plan.auxiliary_outputs, ("aux",))
        expression = GemmReductionDescriptor.parse("mean_linear:1:2:3")
        self.assertEqual(expression.serialize(), "mean_linear:1:2:3")
        expression = GemmReductionDescriptor.parse("custom_reduction:1:2")
        self.assertEqual(expression.serialize(), "custom_reduction:1:2")
        self.assertFalse(expression.has_valid_parameters)

    def test_reduction_pattern_near_misses(self):
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRExpression as Expr,
            GemmEpilogueIRStore,
            is_absmax_normalize_ir,
            is_logsumexp_ir,
            is_softmax_ir,
        )

        load = Expr("load", ("gemm", 0, None))
        scale = Expr("load", ("scale", 0, None))
        one = Expr("constant", (1.0, torch.float32))
        near_softmax = Expr("truediv", (Expr("exp", (load,)), Expr("add", (load, one))))
        near_absmax = Expr(
            "mul", (load, Expr("reciprocal", (Expr("add", (scale, one)),)))
        )
        near_logsumexp = Expr(
            "add", (Expr("log", (Expr("exp", (load,)),)), Expr("maximum", (load, one)))
        )

        self.assertFalse(is_softmax_ir(GemmEpilogueIRStore(0, near_softmax), "gemm", 4))
        self.assertFalse(
            is_absmax_normalize_ir(GemmEpilogueIRStore(0, near_absmax), "gemm", "scale")
        )
        self.assertFalse(
            is_logsumexp_ir(GemmEpilogueIRStore(0, near_logsumexp), "gemm", 4)
        )

    def _create_mock_kernel(self, tile_m, tile_n, tile_k, cluster_m, cluster_n):
        """Create a mock kernel with the given tile/cluster configuration."""
        kernel = MagicMock()
        kernel.metadata.design.tile_shape = (tile_m, tile_n, tile_k)
        kernel.metadata.design.cluster_shape = (cluster_m, cluster_n)
        return kernel

    def _create_mock_inputs(self, m=512, n=512, k=512, dtype=torch.float16):
        """Create a mock MMKernelInputs."""
        inputs = MagicMock()
        inputs.mnk_hinted.return_value = (m, n, k)
        inputs.dtype.return_value = dtype
        inputs._mat1_idx = 0
        inputs._mat2_idx = 1
        inputs.strides_hinted.return_value = ((k, 1), (n, 1))
        return inputs

    def test_fallback_when_heuristics_unavailable(self):
        """Test that filter_kernels returns first N kernels when heuristics unavailable."""
        heuristics = NVUniversalGemmHeuristics()

        kernels = [self._create_mock_kernel(128, 128, 64, 1, 1) for _ in range(10)]
        inputs = self._create_mock_inputs()

        with patch.object(heuristics, "should_run", return_value=False):
            result = heuristics.filter_kernels(kernels, inputs, count=3)

        self.assertEqual(len(result), 3)
        self.assertEqual(result, kernels[:3])

    def test_fallback_when_no_configs_extracted(self):
        """Test fallback when kernel configs cannot be extracted."""
        heuristics = NVUniversalGemmHeuristics()

        kernels = []
        for _ in range(5):
            kernel = MagicMock()
            kernel.metadata.design = MagicMock(spec=[])  # No tile_shape attr
            kernels.append(kernel)

        inputs = self._create_mock_inputs()

        with patch.object(heuristics, "should_run", return_value=True):
            result = heuristics.filter_kernels(kernels, inputs, count=2)

        self.assertEqual(len(result), 2)
        self.assertEqual(result, kernels[:2])

    def test_filter_kernels_sorts_by_runtime(self):
        """Test that filter_kernels returns kernels sorted by estimated runtime and respects count."""
        heuristics = NVUniversalGemmHeuristics()

        kernel_a = self._create_mock_kernel(128, 128, 64, 1, 1)
        kernel_b = self._create_mock_kernel(256, 128, 64, 2, 1)
        kernel_c = self._create_mock_kernel(128, 256, 32, 1, 2)
        kernels = [kernel_a, kernel_b, kernel_c]

        inputs = self._create_mock_inputs()

        heuristic_configs = [
            HeuristicConfig(128, 128, 64, 1, 1, 4, 1, 64, 64, 32, 0.003),
            HeuristicConfig(256, 128, 64, 2, 1, 4, 1, 64, 64, 32, 0.001),
            HeuristicConfig(128, 256, 32, 1, 2, 4, 1, 64, 64, 32, 0.002),
        ]

        with patch.object(heuristics, "should_run", return_value=True):
            with patch.object(
                heuristics, "_get_heuristic_configs", return_value=heuristic_configs
            ):
                # Test sorting with count=3 (all kernels)
                result = heuristics.filter_kernels(kernels, inputs, count=3)
                self.assertEqual(len(result), 3)
                self.assertIs(result[0], kernel_b)
                self.assertIs(result[1], kernel_c)
                self.assertIs(result[2], kernel_a)

                # Test count limit with count=2 (should drop slowest)
                result = heuristics.filter_kernels(kernels, inputs, count=2)
                self.assertEqual(len(result), 2)
                self.assertIs(result[0], kernel_b)
                self.assertIs(result[1], kernel_c)

                # Test count > available kernels (should return all 3)
                result = heuristics.filter_kernels(kernels, inputs, count=10)
                self.assertEqual(len(result), 3)

    def test_filter_kernels_deduplicates_heuristic_configs(self):
        heuristics = NVUniversalGemmHeuristics()
        kernel = self._create_mock_kernel(128, 128, 64, 1, 1)
        kernel.metadata.operator_name = "kernel"
        config = HeuristicConfig(128, 128, 64, 1, 1, 4, 1, 64, 64, 32, 0.003)

        with (
            patch.object(heuristics, "should_run", return_value=True),
            patch.object(
                heuristics, "_get_heuristic_configs", return_value=[config, config]
            ),
        ):
            result = heuristics.filter_kernels(
                [kernel], self._create_mock_inputs(), count=2
            )

        self.assertEqual(result, [kernel])


@unittest.skipIf(
    not (
        ensure_nv_universal_gemm_available()
        and is_datacenter_blackwell_arch()
        and ensure_nvmatmul_heuristics_available()
    ),
    "Requires cutlass_api, nvMatmulHeuristics, and Blackwell GPU",
)
class TestNVUniversalGemmHeuristicsIntegration(TestCase):
    """Integration tests for nvMatmulHeuristics with real library calls."""

    def test_fp4_heuristic_configs(self):
        """Test that nvMatmulHeuristics returns configs for FP4 blockscaled GEMM."""
        heuristics = NVUniversalGemmHeuristics()

        m, n, k = 256, 512, 1024
        configs = heuristics._get_heuristic_configs(
            m,
            n,
            k,
            dtype_a=torch.float4_e2m1fn_x2,
            layout_a="row",
            layout_b="col",
            count=5,
            valid_configs=OrderedSet(),
            accumulator_type=torch.float32,
            dtype_b=torch.float4_e2m1fn_x2,
            out_dtype=torch.float32,
        )

        self.assertGreater(
            len(configs), 0, "nvMatmulHeuristics returned no FP4 configs"
        )
        for cfg in configs:
            self.assertGreater(cfg.tile_m, 0)
            self.assertGreater(cfg.tile_n, 0)
            self.assertGreater(cfg.estimated_runtime, 0)

    def test_fp8_heuristic_configs(self):
        """Test that nvMatmulHeuristics returns configs for FP8 GEMM."""
        heuristics = NVUniversalGemmHeuristics()

        m, n, k = 256, 512, 1024
        configs = heuristics._get_heuristic_configs(
            m,
            n,
            k,
            dtype_a=torch.float8_e4m3fn,
            layout_a="row",
            layout_b="col",
            count=5,
            valid_configs=OrderedSet(),
            accumulator_type=torch.float32,
            dtype_b=torch.float8_e4m3fn,
            out_dtype=torch.float32,
        )

        self.assertGreater(
            len(configs), 0, "nvMatmulHeuristics returned no FP8 configs"
        )
        for cfg in configs:
            self.assertGreater(cfg.tile_m, 0)
            self.assertGreater(cfg.tile_n, 0)
            self.assertGreater(cfg.estimated_runtime, 0)


@unittest.skipIf(
    not (ensure_nv_universal_gemm_available() and is_datacenter_blackwell_arch()),
    "NVIDIA Universal GEMM (cutlass_api) library not available or not on Blackwell",
)
class TestNVUniversalGemmDynamicShapes(TestCase):
    """Test cases for NVIDIA Universal GEMM with dynamic shapes."""

    @torch._dynamo.config.patch({"capture_dynamic_output_shape_ops": True})
    def test_unbacked_symint_rejected(self):
        """Test that NVGEMM rejects unbacked symbolic integers."""

        def fn(x, w):
            nz = torch.nonzero(x)  # Creates unbacked symint for nz.size(0)
            a = torch.ones(nz.size(0), w.size(0), dtype=w.dtype, device=w.device)
            return a @ w

        x = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0], device="cuda")
        w = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")

        torch._dynamo.reset()

        with config.patch(_nvgemm_config(nvgemm_max_profiling_configs=2)):
            compiled_fn = torch.compile(fn, dynamic=True)
            with self.assertRaisesRegex(
                Exception, "NoValidChoicesError|no valid choice"
            ):
                compiled_fn(x, w)

    def test_dynamic_shapes(self):
        """Stress test dynamic shapes with extreme variations."""

        def matmul(a, b):
            return a @ b

        torch._dynamo.reset()

        with config.patch(_nvgemm_config(nvgemm_max_profiling_configs=2)):
            compiled_fn = torch.compile(matmul, dynamic=True)

            shapes = [
                (4, 4, 4, False),
                (16, 16, 16, True),
                (2048, 64, 128, True),
                (4, 4, 4, False),  # Unsupported again
                (64, 2048, 128, True),
                (128, 128, 2048, True),
                (2048, 2048, 512, True),
                (16, 16, 16, True),
            ]

            for m, n, k, supported in shapes:
                a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
                if not supported:
                    with self.assertRaisesRegex(
                        Exception, "NoValidChoicesError|no valid choice"
                    ):
                        compiled_fn(a, b)
                else:
                    result = compiled_fn(a, b)
                    torch.testing.assert_close(result, a @ b)


@unittest.skipIf(
    not (ensure_nv_universal_gemm_available() and is_datacenter_blackwell_arch()),
    "NVIDIA Universal GEMM (cutlass_api) library not available or not on Blackwell",
)
@instantiate_parametrized_tests
class TestNVUniversalGemmEpilogueFusion(TestCase):
    """Test cases for NVIDIA Universal GEMM epilogue fusion.

    Tests verify both correctness and that fusion actually occurs by examining
    generated code for epilogue markers. Benchmarks are mocked to ensure
    deterministic fusion decisions independent of GPU noise.
    """

    M, N, K = 512, 512, 512

    def _compile_and_check(self, fn, *args):
        torch._dynamo.reset()
        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "NVGEMM",
                    "nvgemm_max_profiling_configs": 2,
                    "force_disable_caches": True,
                }
            ),
            mock.patch.object(
                Scheduler,
                "benchmark_fused_nodes",
                return_value=(1.0, ""),
            ),
            mock.patch.object(
                Scheduler,
                "benchmark_codegened_module",
                return_value=(0.5, ""),
            ),
        ):
            result, code_list = run_and_get_code(torch.compile(fn), *args)
        code = "\n".join(code_list)
        self.assertEqual(code.count(".run("), 1)
        epilogue_fused = EPILOGUE_FN_NAME in code and "EpilogueArguments" in code
        return result, code, epilogue_fused

    def _compile_dynamic(self, fn, args, second_args):
        torch._dynamo.reset()
        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "NVGEMM",
                    "nvgemm_max_profiling_configs": 2,
                    "force_disable_caches": True,
                }
            ),
            mock.patch.object(
                Scheduler, "benchmark_fused_nodes", return_value=(1.0, "")
            ),
            mock.patch.object(
                Scheduler, "benchmark_codegened_module", return_value=(0.5, "")
            ),
        ):
            compiled = torch.compile(fn, dynamic=True)
            result, code_list = run_and_get_code(compiled, *args)
            second_result = compiled(*second_args)
        return result, second_result, "\n".join(code_list)

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_matmul_pointwise_epilogue_fusion(self, dtype):
        """Pointwise op (relu) is fused into the GEMM epilogue."""
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)

        def fn(a, b):
            return torch.relu(a @ b)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        torch.testing.assert_close(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "pointwise op was NOT fused into epilogue")

    def test_matmul_multi_store_epilogue_fusion(self):
        a = torch.randn(self.M, self.K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(self.K, self.N, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = (a @ b).float()
            return torch.relu(result), result + 1.0

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        self.assertIn("EpilogueArguments", code)
        self.assertNotIn("CuTeDSLEpilogueArguments", code)
        self.assertEqual(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused)
        self.assertIn("out_ptr1", code)

    def test_flex_gemm_pointwise_epilogue_fusion(self):
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                torch.relu,
                kernel_options={"backend": "NVGEMM"},
            )

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        torch.testing.assert_close(result, torch.relu(a @ b), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "FlexGEMM body was NOT fused into epilogue")

    def test_flex_gemm_addmm_pointwise_epilogue_fusion(self):
        m, n, k = 128, 128, 64
        bias = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(bias, a, b):
            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                torch.relu,
                gemm_kwargs={"beta": 0.5, "alpha": 1.5},
                kernel_options={"backend": "NVGEMM"},
            )

        result, code, _ = self._compile_and_check(fn, bias, a, b)
        expected = torch.addmm(bias, a, b, beta=0.5, alpha=1.5).relu()
        self.assertEqual(result, expected, atol=2e-2, rtol=2e-2)
        self.assertIn("VendoredDenseGemmEFCOperator", code)

    def test_flex_gemm_addmm_beta_zero_ignores_bias(self):
        m, n, k = 128, 128, 64
        bias = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(bias, a, b):
            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                torch.relu,
                gemm_kwargs={"beta": 0, "alpha": 1.5},
                kernel_options={"backend": "NVGEMM"},
            )

        result, code, _ = self._compile_and_check(fn, bias, a, b)
        expected = torch.addmm(bias, a, b, beta=0, alpha=1.5).relu()
        self.assertEqual(result, expected, atol=2e-2, rtol=2e-2)
        self.assertIn("VendoredDenseGemmEFCOperator", code)

    @parametrize("batch", (1, 2))
    def test_flex_gemm_bmm_pointwise_epilogue_fusion(self, batch):
        m, n, k = 128, 128, 64
        a = torch.randn(batch, m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(batch, k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            return flex_gemm(
                torch.bmm,
                (a, b),
                torch.relu,
                kernel_options={"backend": "NVGEMM"},
            )

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, torch.bmm(a, b).relu(), atol=2e-2, rtol=2e-2)
        self.assertIn("VendoredDenseGemmEFCOperator", code)

    @parametrize("bias_kind", ("batch", "tile", "row_1d", "row_2d", "col_2d"))
    def test_flex_gemm_baddbmm_pointwise_epilogue_fusion(self, bias_kind):
        batch, m, n, k = 2, 128, 192, 64
        bias_shapes = {
            "batch": (batch, m, n),
            "tile": (m, n),
            "row_1d": (n,),
            "row_2d": (1, n),
            "col_2d": (m, 1),
        }
        bias = torch.randn(*bias_shapes[bias_kind], device="cuda", dtype=torch.bfloat16)
        a = torch.randn(batch, m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(batch, k, n, device="cuda", dtype=torch.bfloat16)

        def fn(bias, a, b):
            return flex_gemm(
                torch.baddbmm,
                (bias, a, b),
                torch.relu,
                gemm_kwargs={"beta": 0.5, "alpha": 1.5},
                kernel_options={"backend": "NVGEMM"},
            )

        result, code, _ = self._compile_and_check(fn, bias, a, b)
        expected = torch.baddbmm(bias, a, b, beta=0.5, alpha=1.5).relu()
        self.assertEqual(result, expected, atol=2e-2, rtol=2e-2)
        self.assertIn("VendoredDenseGemmEFCOperator", code)

    def test_flex_gemm_dynamic_tuple_output_fusion(self):
        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: (
                    ((acc.float() + 1.0) * 0.5).to(torch.bfloat16),
                    acc.float().square() + 2.0,
                ),
                kernel_options={"backend": "NVGEMM"},
            )

        torch._dynamo.reset()
        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "NVGEMM",
                    "nvgemm_max_profiling_configs": 2,
                    "force_disable_caches": True,
                }
            ),
            mock.patch.object(
                Scheduler,
                "benchmark_fused_nodes",
                return_value=(1.0, ""),
            ),
            mock.patch.object(
                Scheduler,
                "benchmark_codegened_module",
                return_value=(0.5, ""),
            ),
        ):
            compiled = torch.compile(fn, dynamic=True)
            a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
            result, code_list = run_and_get_code(compiled, a, b)
            self.assertEqual(result, fn(a, b), atol=2e-2, rtol=2e-2)
            a2 = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
            b2 = torch.randn(64, 192, device="cuda", dtype=torch.bfloat16)
            self.assertEqual(compiled(a2, b2), fn(a2, b2), atol=2e-2, rtol=2e-2)
        self.assertIn("VendoredDenseGemmEFCOperator", "\n".join(code_list))

    @parametrize("shape", ((128, 128, 64), (144, 96, 32)))
    def test_bf16_tuple_aux_bool_output_fusion(self, shape):
        m, n, k = shape
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = a @ b
            return torch.relu(result), result.float() > 0

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b))
        self.assertEqual(result[1].dtype, torch.bool)
        self.assertIn("VendoredDenseGemmEFCOperator", code)
        self.assertIn("reduction_type='direct_bool_gt_zero'", code)

    def test_bf16_tuple_aux_supports_distinct_output_dtypes(self):
        m, n, k = 128, 128, 64
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = a @ b
            return torch.relu(result), result.float().square() + 2.0

        result, code, _ = self._compile_and_check(fn, a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected, atol=2e-2, rtol=2e-2)
        self.assertEqual(result[0].dtype, torch.bfloat16)
        self.assertEqual(result[1].dtype, torch.float32)
        self.assertIn("VendoredDenseGemmEFCOperator", code)

    @parametrize("feeds_main", (False, True))
    def test_flex_gemm_grouped_n_reduce_epilogue_fusion(self, feeds_main):
        m, n, k, group = 128, 128, 64, 16
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        def epilogue(acc):
            grouped = acc.float().view(m, -1, group)
            sums = grouped.sum(-1, keepdim=True)
            if feeds_main:
                return (grouped / sums).reshape(m, n).to(torch.bfloat16)
            return torch.relu(acc), sums.squeeze(-1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "NVGEMM"},
            )

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, epilogue(a @ b))
        self.assertIn("VendoredDenseGemmEFCOperator", code)
        self.assertIn("group=16", code)
        self.assertIn(f"feeds_main={feeds_main}", code)

    @parametrize("chunked", (False, True))
    def test_flex_gemm_generic_output_contraction_falls_back(self, chunked):
        m, n, k = 128, 128, 64
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, 2 * n, device="cuda", dtype=torch.bfloat16)

        def epilogue(acc):
            value = acc.float()
            if chunked:
                gate, up = value.chunk(2, dim=-1)
            else:
                pairs = value.view(m, n, 2)
                gate, up = pairs[..., 0], pairs[..., 1]
            return (torch.nn.functional.silu(gate) * up).to(acc.dtype)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "NVGEMM"},
            )

        result, _, epilogue_fused = self._compile_and_check(
            fn, a, b, expected_kernels=2
        )
        self.assertEqual(result, epilogue(a @ b), atol=2e-2, rtol=2e-2)
        self.assertFalse(epilogue_fused)

    @parametrize(
        "case",
        (
            (4, "sum"),
            (16, "sum"),
            (16, "mean"),
            (16, "prod"),
            (16, "amax"),
            (16, "amin"),
            (32, "sum"),
            (64, "sum"),
            (64, "mean"),
            (64, "amax"),
            (128, "sum"),
            (16, "square_mean"),
            (16, "abs_amax"),
        ),
        name_fn=lambda case: f"group_{case[0]}_{case[1]}",
    )
    def test_bf16_grouped_n_reduce_epilogue_fusion(self, case):
        group, reduction = case
        m, n, k = 128, 256 if group == 128 else 128, 64
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            source, _, reduce_op = reduction.rpartition("_")
            if source == "square":
                grouped = grouped.square()
            elif source == "abs":
                grouped = grouped.abs()
            reduced = getattr(grouped, reduce_op)(-1)
            return torch.relu(result), reduced

        supported = (group, reduction) != (64, "mean")
        result, code, _ = self._compile_and_check(
            fn, a, b, expected_kernels=1 if supported else None
        )
        expected = fn(a, b)
        self.assertEqual(result[1], expected[1], atol=1e-2, rtol=1e-2)
        self.assertEqual(result[0], expected[0], atol=1e-2, rtol=1e-2)
        if supported:
            self.assertIn("VendoredDenseGemmEFCOperator", code)
            self.assertIn("output=", code)
            self.assertIn(f"group={group}", code)
        else:
            self.assertNotIn("reduction_type='mean'", code)

    @parametrize(
        "case",
        (
            (2, "sum"),
            (2, "mean"),
            (2, "prod"),
            (2, "amax"),
            (2, "amin"),
            (16, "sum"),
            (16, "mean"),
            (16, "prod"),
            (16, "amax"),
            (16, "amin"),
            (32, "sum"),
            (32, "mean"),
            (32, "prod"),
            (32, "amax"),
            (32, "amin"),
            (64, "sum"),
            (64, "mean"),
            (64, "amax"),
            (128, "sum"),
            (128, "mean"),
            (128, "amax"),
            (16, "square_mean"),
            (64, "abs_amax"),
        ),
        name_fn=lambda case: f"group_{case[0]}_{case[1]}",
    )
    def test_bf16_grouped_m_reduce_epilogue_fusion(self, case):
        group, reduction = case
        m, n, k = 128, 128, 64
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(-1, group, n)
            source, _, reduce_op = reduction.rpartition("_")
            if source == "square":
                grouped = grouped.square()
            elif source == "abs":
                grouped = grouped.abs()
            reduced = getattr(grouped, reduce_op)(1)
            return torch.relu(result), reduced

        result, code, _ = self._compile_and_check(fn, a, b)
        expected = fn(a, b)
        self.assertEqual(result[1], expected[1], atol=1e-2, rtol=1e-2)
        self.assertEqual(result[0], expected[0], atol=1e-2, rtol=1e-2)
        self.assertIn("VendoredDenseGemmEFCOperator", code)
        self.assertIn("axis=0", code)
        self.assertIn(f"group={group}", code)

    def test_bf16_dynamic_grouped_n_reduce_epilogue_fusion(self):
        n, k = 128, 64
        a = torch.randn(128, k, device="cuda", dtype=torch.bfloat16)
        a2 = torch.randn(192, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(result.shape[0], -1, 16)
            return torch.relu(result), grouped.sum(-1)

        result, result2, code = self._compile_dynamic(fn, (a, b), (a2, b))
        expected = fn(a, b)
        expected2 = fn(a2, b)
        self.assertEqual(result[0], expected[0], atol=1e-2, rtol=1e-2)
        self.assertEqual(result[1], expected[1], atol=1e-2, rtol=1e-2)
        self.assertEqual(result2[0], expected2[0], atol=1e-2, rtol=1e-2)
        self.assertEqual(result2[1], expected2[1], atol=1e-2, rtol=1e-2)
        self.assertIn("VendoredDenseGemmEFCOperator", code)
        self.assertIn("group=16", code)

    @parametrize("reduction", ("amax", "amin"))
    def test_bf16_grouped_n_extrema_preserve_nan(self, reduction):
        m, n, k, group = 16, 128, 16, 64
        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        b[0, 0] = float("nan")

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            return result, getattr(grouped, reduction)(-1)

        result, code, _ = self._compile_and_check(fn, a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected, equal_nan=True)
        self.assertTrue(torch.isnan(result[1][:, 0]).all())
        self.assertEqual(result[1][:, 1], torch.zeros_like(result[1][:, 1]))
        self.assertIn("group=64", code)

    def test_bf16_grouped_m_reduce_supports_post_reduction_pointwise(self):
        m, n, k, group = 128, 128, 64, 64
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(-1, group, n)
            reduced = (grouped.abs().amax(1) + 1.0).sqrt()
            return torch.relu(result), reduced

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertIn("VendoredDenseGemmEFCOperator", code)
        self.assertIn("axis=0", code)
        self.assertIn("group=64", code)

    def test_bf16_grouped_n_reduce_supports_post_reduction_pointwise(self):
        m, n, k, group = 128, 128, 64, 64
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            return torch.relu(result), grouped.sum(-1) + 1.0

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertIn("VendoredDenseGemmEFCOperator", code)
        self.assertIn("axis=1", code)
        self.assertIn("group=64", code)

    def test_bf16_dynamic_grouped_m_reduce_epilogue_fusion(self):
        n, k, group = 128, 64, 16
        a = torch.randn(128, k, device="cuda", dtype=torch.bfloat16)
        a2 = torch.randn(192, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(result.shape[0] // group, group, n)
            return torch.relu(result), grouped.sum(1)

        result, result2, code = self._compile_dynamic(fn, (a, b), (a2, b))
        self.assertEqual(result, fn(a, b))
        self.assertEqual(result2, fn(a2, b))
        self.assertIn("VendoredDenseGemmEFCOperator", code)
        self.assertIn("axis=0", code)
        self.assertIn("group=16", code)

    def test_bf16_grouped_m_reduce_supports_cta_tail(self):
        m, n, k, group = 144, 128, 64, 16
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(-1, group, n)
            return torch.relu(result), grouped.sum(1)

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=2e-2, rtol=2e-2)
        self.assertIn("axis=0", code)
        self.assertIn("group=16", code)

    def test_bf16_grouped_m_reduce_transposed_b_input(self):
        m, n, k, group = 128, 192, 64, 16
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b.mT
            grouped = result.float().view(-1, group, n)
            return torch.relu(result), grouped.sum(1)

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=2e-2, rtol=2e-2)
        self.assertIn("axis=0", code)
        self.assertIn("group=16", code)

    def test_bf16_grouped_n_reduce_supports_tuple_reshape(self):
        m, n, k, group = 128, 128, 64, 16
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            return torch.relu(result), grouped.sum(-1).view(8, 16, 8)

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=2e-2, rtol=2e-2)
        self.assertIn("axis=1", code)
        self.assertIn("group=16", code)

    def test_bf16_grouped_n_chained_logsumexp_fusion(self):
        m, n, k, group = 128, 96, 64, 4
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            return torch.relu(result), grouped.logsumexp(-1)

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=2e-2, rtol=2e-2)
        self.assertIn("reduction_type='logsumexp'", code)
        self.assertIn("VendoredDenseGemmEFCOperator", code)

    def test_bf16_grouped_n_reduction_near_miss_falls_back(self):
        m, n, k, group = 128, 96, 64, 4
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            reduced = grouped.exp().sum(-1).log() + grouped.amax(-1)
            return torch.relu(result), reduced

        result, code, _ = self._compile_and_check(fn, a, b, expected_kernels=2)
        self.assertEqual(result, fn(a, b), atol=2e-2, rtol=2e-2)
        self.assertNotIn("reduction_type='logsumexp'", code)

    def test_bf16_grouped_n_chained_logsumexp_special_values(self):
        m, n, k, group = 8, 96, 8, 4
        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        b[0, 0] = float("inf")
        b[:, 4:8] = float("-inf")
        b[0, 8] = float("nan")

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            return torch.relu(result), grouped.logsumexp(-1)

        result, code, _ = self._compile_and_check(fn, a, b)
        expected = fn(a, b)
        self.assertEqual(result[1], expected[1], equal_nan=True)
        self.assertTrue(torch.isposinf(result[1][:, 0]).all())
        self.assertTrue(torch.isneginf(result[1][:, 1]).all())
        self.assertTrue(torch.isnan(result[1][:, 2]).all())
        self.assertIn("reduction_type='logsumexp'", code)

    def test_bf16_grouped_n_chained_stable_logsumexp_fusion(self):
        m, n, k, group = 128, 96, 64, 4
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            maximum = grouped.amax(-1, keepdim=True)
            reduced = (grouped - maximum).exp().sum(-1).log()
            return torch.relu(result), reduced + maximum.squeeze(-1)

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=2e-2, rtol=2e-2)
        self.assertIn("reduction_type='logsumexp'", code)

    @parametrize("affine", ((0.5, 1.0), (2.0, -0.25)))
    def test_bf16_grouped_n_chained_variance_fusion(self, affine):
        m, n, k, group = 128, 96, 64, 4
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1
        scale, bias = affine

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            centered = grouped - grouped.mean(-1, keepdim=True)
            variance = centered.square().mean(-1) * scale + bias
            return torch.relu(result), variance

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=2e-2, rtol=2e-2)
        self.assertIn("reduction_type='variance_affine:", code)
        self.assertIn("VendoredDenseGemmEFCOperator", code)

    @parametrize(
        "case",
        (
            (2, "forward"),
            (16, "affine"),
            (16, "reverse"),
            (16, "with_sum"),
            (16, "centered"),
            (8, "centered_aux"),
            (8, "centered_with_mean"),
            (16, "reverse_centered"),
            (16, "mean_affine"),
            (32, "forward"),
            (8, "trailing"),
            (32, "fp8"),
            (32, "fp8_with_sum"),
        ),
        name_fn=lambda case: f"group_{case[0]}_{case[1]}",
    )
    def test_bf16_grouped_m_sum_normalize_epilogue_fusion(self, case):
        group, mode = case
        m, n, k = 128, 128, 64
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)
        mean_mode = mode in (
            "centered",
            "centered_aux",
            "centered_with_mean",
            "reverse_centered",
            "mean_affine",
        )

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(-1, group, n)
            reductions = (
                grouped.mean(1, keepdim=True)
                if mean_mode
                else grouped.sum(1, keepdim=True)
            )
            if mode in ("centered", "centered_aux", "centered_with_mean"):
                normalized = grouped - reductions
            elif mode == "reverse_centered":
                normalized = reductions - grouped
            elif mode == "mean_affine":
                normalized = grouped * 2.0 + reductions * 3.0 + 0.5
            elif mode == "affine":
                normalized = grouped / (reductions * 2.0 + 1.0) * 3.0 + 0.5
            elif mode == "reverse":
                normalized = (reductions + 1.0) / grouped
            else:
                normalized = grouped / reductions
            if mode == "trailing":
                normalized = normalized * 0.5 + 1.0
            if mode != "centered_aux":
                normalized = normalized.to(
                    torch.float8_e4m3fn
                    if mode in ("fp8", "fp8_with_sum")
                    else torch.bfloat16
                )
            if mode == "centered_aux":
                return torch.relu(result), normalized
            return (
                (normalized, reductions.squeeze(1))
                if mode in ("with_sum", "centered_with_mean", "fp8_with_sum")
                else normalized
            )

        result, code, _ = self._compile_and_check(fn, a, b)
        expected = fn(a, b)
        if mode == "fp8_with_sum":
            self.assertEqual(
                result[0].float(), expected[0].float(), atol=2e-2, rtol=2e-2
            )
            self.assertEqual(result[1], expected[1], atol=2e-2, rtol=2e-2)
        elif mode == "fp8":
            self.assertEqual(result.float(), expected.float(), atol=2e-2, rtol=2e-2)
        else:
            self.assertEqual(result, expected, atol=2e-2, rtol=2e-2)
        self.assertIn("VendoredDenseGemmEFCOperator", code)
        self.assertIn("axis=0", code)
        self.assertIn("feeds_main=True", code)
        if mean_mode:
            self.assertIn("mean_linear:", code)
        if mode in ("fp8", "fp8_with_sum"):
            output = result[0] if isinstance(result, tuple) else result
            self.assertEqual(output.dtype, torch.float8_e4m3fn)

    def test_bf16_grouped_m_outputs_share_physical_reduction(self):
        m, n, k, group = 128, 128, 64, 8
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(-1, group, n)
            scale = grouped.sum(1, keepdim=True)
            normalized = (grouped / scale).view(m, n).to(torch.bfloat16)
            scaled = (grouped * (scale + 1.0)).view(m, n)
            return normalized, scaled

        result, code, _ = self._compile_and_check(fn, a, b, expected_kernels=2)
        self.assertIn("VendoredDenseGemmEFCOperator", code)
        self.assertIn("axis=0", code)
        self.assertIn("feeds_main=False", code)
        self.assertIn("output=out_ptr1", code)
        self.assertEqual(result, fn(a, b), atol=2e-2, rtol=2e-2)

    def test_bf16_grouped_m_regrouped_reduction_reuse(self):
        m, n, k, group = 128, 64, 64, 8
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(-1, group, n)
            scale = grouped.sum(1, keepdim=True)
            normalized = (grouped * scale.reciprocal()).view(m, n)
            regrouped = grouped.view(m, n).view(-1, group, n)
            aux = (regrouped * scale.reciprocal()).view(m, n)
            return normalized, aux

        result, code, _ = self._compile_and_check(fn, a, b, expected_kernels=2)
        self.assertEqual(result, fn(a, b), atol=2e-2, rtol=2e-2)
        self.assertIn("VendoredDenseGemmEFCOperator", code)
        self.assertIn("feeds_main=False", code)
        self.assertIn("output=out_ptr1", code)

    def test_bf16_grouped_m_composite_reduction_falls_back(self):
        m, n, k, group = 128, 64, 64, 8
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(-1, group, n)
            scale = grouped.sum(1, keepdim=True) + grouped.amax(1, keepdim=True)
            return (grouped / scale).view(m, n)

        result, code, _ = self._compile_and_check(fn, a, b, expected_kernels=3)
        self.assertEqual(result, fn(a, b), atol=2e-2, rtol=2e-2)
        self.assertNotIn("feeds_main=True", code)

    def test_bf16_grouped_m_mixed_layouts_fall_back(self):
        m, n, k = 128, 64, 64
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = a @ b
            grouped8 = result.float().view(-1, 8, n)
            grouped4 = result.float().view(-1, 4, n)
            out = grouped8 / grouped8.sum(1, keepdim=True)
            aux = grouped4 / grouped4.sum(1, keepdim=True)
            return out.view(m, n), aux.view(m, n)

        result, code, _ = self._compile_and_check(fn, a, b, expected_kernels=3)
        self.assertEqual(result, fn(a, b), atol=1e-1, rtol=2e-2)
        self.assertNotIn("feeds_main=True", code)

    def test_bf16_grouped_m_hidden_reduction_falls_back(self):
        m, n, k, group = 128, 64, 64, 8
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(-1, group, n)
            relu_grouped = result.float().relu().view(-1, group, n)
            scaled = grouped / (grouped.sum(1, keepdim=True) + 1.0)
            return (scaled + relu_grouped.sum(1, keepdim=True)).view(m, n)

        result, code, _ = self._compile_and_check(fn, a, b, expected_kernels=3)
        self.assertEqual(result, fn(a, b), atol=1e-1, rtol=2e-2)
        self.assertNotIn("feeds_main=True", code)

    def test_bf16_grouped_m_repeated_reductions_fall_back(self):
        m, n, k, group = 128, 64, 64, 8
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = a @ b
            grouped0 = result.float().view(-1, group, n)
            out = grouped0 / grouped0.sum(1, keepdim=True)
            grouped1 = result.float().view(-1, group, n)
            aux = grouped1 * (grouped1.sum(1, keepdim=True) + 1.0)
            return out.view(m, n), aux.view(m, n)

        result, code, _ = self._compile_and_check(fn, a, b, expected_kernels=3)
        self.assertEqual(result, fn(a, b), atol=1e-1, rtol=2e-2)
        self.assertNotIn("feeds_main=True", code)

    @parametrize(
        "case",
        (
            (4, "forward"),
            (16, "affine"),
            (32, "reverse"),
            (64, "forward"),
            (16, "with_sum"),
            (32, "fp8"),
            (16, "fp8_with_sum"),
        ),
        name_fn=lambda case: f"group_{case[0]}_{case[1]}",
    )
    def test_bf16_grouped_n_sum_normalize_epilogue_fusion(self, case):
        group, mode = case
        m, n, k = 128, 128, 64
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            sums = grouped.sum(-1, keepdim=True)
            if mode == "affine":
                normalized = grouped / (sums * 2.0 + 1.0) * 3.0 + 0.5
            elif mode == "reverse":
                normalized = (sums + 1.0) / grouped
            else:
                normalized = grouped / sums
            normalized = normalized.to(
                torch.float8_e4m3fn
                if mode in ("fp8", "fp8_with_sum")
                else torch.bfloat16
            )
            if mode in ("with_sum", "fp8_with_sum"):
                return normalized, sums.squeeze(-1)
            return normalized

        supported = group <= 32
        result, code, _ = self._compile_and_check(
            fn, a, b, expected_kernels=1 if supported else None
        )
        expected = fn(a, b)
        if mode == "fp8_with_sum":
            self.assertEqual(
                result[0].float(), expected[0].float(), atol=1e-2, rtol=1e-2
            )
            self.assertEqual(result[1], expected[1], atol=1e-2, rtol=1e-2)
        elif mode == "fp8":
            self.assertEqual(result.float(), expected.float(), atol=1e-2, rtol=1e-2)
        else:
            self.assertEqual(result, expected, atol=1e-2, rtol=1e-2)
        if supported:
            self.assertIn("VendoredDenseGemmEFCOperator", code)
            self.assertIn("feeds_main=True", code)
            reduce_type = (
                "normalize_sum_reverse_affine"
                if mode == "reverse"
                else "normalize_sum_affine"
            )
            self.assertIn(reduce_type, code)

    def test_flex_gemm_preserves_output_for_unfused_reduction(self):
        dtype = torch.bfloat16
        a = torch.randn(128, 64, device="cuda", dtype=dtype)
        b = torch.randn(64, 128, device="cuda", dtype=dtype)

        def fn(a, b):
            def epilogue(acc):
                grouped = acc.float().view(128, -1, 32)
                return acc.relu(), grouped.sum(-1)

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "NVGEMM"},
            )

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        expected = fn(a, b)
        self.assertEqual(result[0], expected[0], atol=1e-2, rtol=1e-2)
        self.assertEqual(result[1], expected[1], atol=1e-1, rtol=1e-2)
        self.assertTrue(epilogue_fused, "pointwise consumer was NOT fused")
        self.assertIn("out_ptr1", code)

    def test_matmul_add_relu_chained(self):
        """Multi-op pointwise chain (a@b + bias → relu) collapses to one
        ComputedBuffer and is fused as a single epilogue."""
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)
        bias = torch.randn(self.M, self.N, device="cuda", dtype=dtype)

        def fn(a, b, bias):
            return torch.relu((a @ b) + bias)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b, bias)
        torch.testing.assert_close(result, fn(a, b, bias), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "bias+relu chain was NOT fused into epilogue")

    def test_matmul_cast_dtype(self):
        """Output-dtype cast in the epilogue."""
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)

        def fn(a, b):
            return (a @ b).to(torch.float32)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        torch.testing.assert_close(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "dtype cast was NOT fused into epilogue")

    def test_plain_matmul_no_epilogue(self):
        """Test that plain matmul does NOT produce epilogue fusion markers."""
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)

        def fn(a, b):
            return a @ b

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        torch.testing.assert_close(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertFalse(
            epilogue_fused, "plain matmul should NOT have epilogue fusion markers"
        )

    @unittest.skip(
        "Disabled due to CI failures; see "
        "https://github.com/pytorch/pytorch/issues/190235"
    )
    def test_reduction_not_fused(self):
        """Test that reductions after GEMM are NOT fused into the epilogue."""
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)

        def fn(a, b):
            return (a @ b).sum(dim=-1)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        expected = fn(a, b)
        torch.testing.assert_close(
            result.float(), expected.float(), atol=5e-2, rtol=5e-2
        )
        self.assertFalse(
            epilogue_fused,
            "reduction after GEMM should NOT be fused into NVGEMM epilogue",
        )

    def test_epilogue_fusion_eliminates_intermediate(self):
        """Verify that epilogue fusion eliminates the intermediate GEMM buffer.

        When relu is fused, the GEMM output should not be allocated separately.
        The fused kernel writes directly to the final output. We check that
        only one output buffer is allocated for the fused kernel, not two
        (one for GEMM + one for relu)."""
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)

        def fn(a, b):
            return torch.relu(a @ b)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        torch.testing.assert_close(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "relu was NOT fused into epilogue")
        # The fused kernel's main function should take exactly 3 positional
        # params (in_ptr0, in_ptr1, out_ptr0) plus stream. No extra intermediate.
        for line in code.split("\n"):
            if "_main(" in line and "def " in line and "nv_universal_gemm" in line:
                self.assertIn(
                    "in_ptr0, in_ptr1, out_ptr0, stream=None",
                    line,
                    lambda msg: f"{msg}\nUnexpected kernel signature: {line.strip()}",
                )

    def test_epilogue_with_aux_input(self):
        """Epilogue that reads an auxiliary tensor (bias) gets it as a kernel arg."""
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)
        bias = torch.randn(self.M, self.N, device="cuda", dtype=dtype)

        def fn(a, b, bias):
            return torch.relu((a @ b) + bias)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b, bias)
        torch.testing.assert_close(result, fn(a, b, bias), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "bias+relu was NOT fused into epilogue")

    @parametrize(
        "bias_shape",
        ((N,), (1, N), (M, 1)),
        name_fn=lambda shape: "x".join(map(str, shape)),
    )
    def test_epilogue_with_broadcast_aux_input(self, bias_shape):
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)
        bias = torch.randn(bias_shape, device="cuda", dtype=dtype)

        def fn(a, b, bias):
            return torch.relu((a @ b) + bias)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b, bias)
        torch.testing.assert_close(result, fn(a, b, bias), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "broadcast bias+relu was not fused")

    def test_epilogue_with_fp32_row_scale(self):
        a = torch.randn(self.M, self.K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(self.K, self.N, device="cuda", dtype=torch.bfloat16)
        scale = torch.rand(self.M, 1, device="cuda", dtype=torch.float32)

        def fn(a, b, scale):
            return (a @ b).float() * scale

        result, _, epilogue_fused = self._compile_and_check(fn, a, b, scale)
        self.assertEqual(result, fn(a, b, scale), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused)

    def test_efc_disk_cache_round_trip(self):
        """Verify that EFC kernel compiled artifacts can be serialized to disk
        and reloaded correctly.

        EFC kernels produce a closure-wrapped compiled_obj. The disk cache
        unwraps the inner JIT function for serialization and rewraps it on
        reload. This test compiles an EFC kernel, round-trips through disk
        cache, and verifies the reloaded artifact produces correct results."""
        from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
            _get_kernel_cache,
        )
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            _rewrap_efc_compiled_obj,
            _unwrap_efc_compiled_obj,
        )
        from torch._inductor.runtime.cutedsl_cache import disk_cache_get, disk_cache_set

        cache = _get_kernel_cache()
        efc_kernel = None
        for name, k in cache.items():
            if (
                "EFC" in name
                and "ABFloat16" in name
                and "outBFloat16" in name
                and "ttt" in name
            ):
                efc_kernel = k
                break
        if efc_kernel is None:
            self.skipTest("No matching EFC kernel found in cache")

        import cutlass_api
        from cutlass_api.artifact import CompiledArtifact

        a = torch.randn(self.M, self.K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(self.K, self.N, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(self.M, self.N, device="cuda", dtype=torch.bfloat16)

        args = cutlass_api.arguments.GemmArguments(
            a, b, out, accumulator_type=torch.float32
        )
        artifact = efc_kernel.compile(args)

        # Unwrap, serialize, reload, rewrap
        inner = _unwrap_efc_compiled_obj(artifact.compiled_obj)
        self.assertNotEqual(
            type(inner).__name__,
            "function",
            "unwrap should extract the JIT function, not the closure",
        )

        test_cache: dict = {}
        disk_cache_set(test_cache, "/tmp/test_efc_rt.py", ("efc",), ("key",), inner, 0)
        loaded = disk_cache_get({}, "/tmp/test_efc_rt.py", ("efc",), ("key",), 0)
        self.assertIsNotNone(loaded, "disk_cache_get returned None")

        rewrapped = _rewrap_efc_compiled_obj(loaded, efc_kernel)
        reloaded_artifact = CompiledArtifact(rewrapped, efc_kernel)

        # Run with reloaded artifact and verify correctness
        out2 = torch.empty(self.M, self.N, device="cuda", dtype=torch.bfloat16)
        args2 = cutlass_api.arguments.GemmArguments(
            a, b, out2, accumulator_type=torch.float32
        )
        efc_kernel.run(
            args2,
            reloaded_artifact,
            stream=torch.cuda.current_stream(),
            workspace=None,
            assume_supported_args=True,
        )
        torch.cuda.synchronize()
        expected = a.float() @ b.float()
        torch.testing.assert_close(out2.float(), expected, atol=1e-2, rtol=1e-2)

    @unittest.skip(
        "Disabled due to CI failures; see "
        "https://github.com/pytorch/pytorch/issues/190234"
    )
    def test_workspace_runtime_integration(self):
        """End-to-end: mock the chosen kernel's workspace_size to non-zero and
        actually let benchmark_codegened_module run, exercising the runtime
        path that consumes the generated get_args()/call() helpers.

        Without the workspace fix, the rendered call would TypeError on the
        missing positional arg, _benchmark_nvgemm_module's broad except would
        return inf, autotune would silently fall back to a non-EFC choice,
        and a naive `assert_close` against eager would still pass. To actually
        catch the regression we intercept _benchmark_nvgemm_module to record
        every (ms, path) it returns and assert at least one finite-ms result —
        i.e., at least one EFC choice with workspace did get benchmarked."""
        import cutlass_api

        from torch._inductor.codegen.cuda_combined_scheduling import (
            CUDACombinedScheduling,
        )

        a = torch.randn(self.M, self.K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(self.K, self.N, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            return torch.relu(a @ b)

        bench_results: list[tuple[float, str]] = []
        orig_bench = CUDACombinedScheduling._benchmark_nvgemm_module

        def capturing_bench(self, module):
            ms, path = orig_bench(self, module)
            bench_results.append((ms, path))
            return ms, path

        torch._dynamo.reset()
        with (
            patch.object(
                cutlass_api.Kernel, "get_workspace_size", lambda self, args: 4096
            ),
            mock.patch.object(
                CUDACombinedScheduling, "_benchmark_nvgemm_module", capturing_bench
            ),
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "NVGEMM",
                    "nvgemm_max_profiling_configs": 2,
                    "force_disable_caches": True,
                }
            ),
        ):
            result = torch.compile(fn)(a, b)
        torch.testing.assert_close(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertTrue(bench_results, "_benchmark_nvgemm_module never invoked")
        finite = [(ms, p) for ms, p in bench_results if ms != float("inf")]
        self.assertTrue(
            finite,
            lambda msg: f"{msg}\nAll NVGEMM benchmarks returned inf — workspace handling likely "
            f"broken. Results: {bench_results}",
        )


if __name__ == "__main__":
    run_tests()
