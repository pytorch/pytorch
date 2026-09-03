# Owner(s): ["module: inductor"]


import threading
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import sympy

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
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    dtype_name,
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
)
from torch.utils._ordered_set import OrderedSet
from torch.utils._triton import has_triton_reduction_ordering


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

    def test_direct_epilogue_cache_signature_distinguishes_wrapped_tensors(self):
        from cutlass.operators.utils.tensor import TensorWrapper

        from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
            _epilogue_args_signature,
        )
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            CuTeDSLEpilogueArguments,
        )

        source = "def epilogue(accum, D):\n    return D"

        def signature(tensor, alignment_bytes=16, *, preserve_layout=False):
            args = CuTeDSLEpilogueArguments(source, D=tensor)
            args.to_tensor_wrappers()
            if preserve_layout or alignment_bytes != 16:
                args.tensors["D"] = TensorWrapper(tensor, alignment_bytes)
            return _epilogue_args_signature(args)

        contiguous = torch.empty(4, 8, device="cuda", dtype=torch.bfloat16)
        transposed = torch.empty(8, 4, device="cuda", dtype=torch.bfloat16).T
        different_shape = torch.empty(8, 8, device="cuda", dtype=torch.bfloat16)
        different_dtype = contiguous.float()
        scalar = torch.empty((), device="cuda", dtype=torch.bfloat16)
        scalar_physical_shape = torch.empty(8, 1, device="cuda", dtype=torch.bfloat16)

        signatures = {
            signature(contiguous, alignment_bytes=4),
            signature(transposed, preserve_layout=True),
            *(
                signature(tensor)
                for tensor in (
                    contiguous,
                    different_shape,
                    different_dtype,
                    scalar,
                    scalar_physical_shape,
                )
            ),
        }
        self.assertEqual(len(signatures), 7)

    def test_nvgemm_cache_key_distinguishes_epilogue_source(self):
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            _create_gemm_cache_key,
            _make_disk_config_key,
        )

        inputs = (torch.empty_strided((4, 4), (4, 1)),) * 2
        out = torch.empty_strided((4, 4), (4, 1))
        kwargs = {"has_epilogue": True, "aux_tensors": ()}
        self.assertNotEqual(
            _create_gemm_cache_key(inputs, out, epilogue_source="first", **kwargs),
            _create_gemm_cache_key(inputs, out, epilogue_source="second", **kwargs),
        )
        self.assertNotEqual(
            _make_disk_config_key(
                "kernel", "GEMM", torch.float32, epilogue_source="first"
            ),
            _make_disk_config_key(
                "kernel", "GEMM", torch.float32, epilogue_source="second"
            ),
        )

    def test_direct_epilogue_schema_separates_local_reduce_result(self):
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            CuTeDSLEpilogueArguments,
        )

        args = CuTeDSLEpilogueArguments(
            "def epilogue(accum, D):\n"
            "    _local_reduce = accum\n"
            "    return D, _local_reduce",
            D=torch.empty(1),
        )
        schema = args.epilogue_fn.schema
        self.assertEqual(schema.outputs, ("D",))
        self.assertEqual(schema.parameter_names, ("D",))
        self.assertTrue(schema.returns_local_reduce)

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
    def test_vendored_dense_grouped_n_feed_main_filters_cta_shapes(self):
        from cutlass import Float32

        from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
            _args_query_candidates,
        )
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            _create_gemm_arguments,
        )
        from torch._inductor.kernel.gemm_epilogue import GemmReductionArguments

        m, n, k = 128, 128, 64
        a = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.empty((k, n), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        feed = torch.empty((m, n), device="cuda")
        args = _create_gemm_arguments(
            "GEMM",
            (a, b),
            out,
            Float32,
            local_reduce=GemmReductionArguments(
                group=16,
                axis=1,
                reduction_type="sum",
                feeds_main=True,
                feed_output=feed,
                consumer_fn=(
                    "def consumer(accumulator, reduction, _secondary):\n"
                    "    return accumulator + reduction"
                ),
            ),
        )
        kernels = [
            kernel
            for kernel in _args_query_candidates(args, 100, efc_only=True)
            if "VendoredDenseGemmEFCOperator" in kernel.metadata.operator_name
        ]

        self.assertTrue(kernels)
        for kernel in kernels:
            design = kernel.metadata.design
            self.assertFalse(design.use_2cta_mma)
            self.assertEqual(design.tile_shape[0], 128)

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

    def test_efc_epilogue_cache_separates_reduction_specializations(self):
        from torch._inductor.codegen.nv_universal_gemm import kernel_cache

        class FakeOperator:
            def __init__(self, metadata):
                self.metadata = metadata

        base_metadata = mock.Mock(
            operands=object(),
            design=object(),
            operator_name="test_efc",
            operator_class=FakeOperator,
            supported_targets=object(),
        )
        base_kernel = FakeOperator(base_metadata)
        epilogue_args = mock.Mock(tensors={})
        sum_specialization = (("reduction_type", "sum"),)
        max_specialization = (("reduction_type", "max"),)

        kernel_cache.clear_cache()
        try:
            with (
                mock.patch.object(
                    kernel_cache, "_meta_epilogue_metadata", return_value=object()
                ),
                mock.patch(
                    "cutlass.operators.metadata.OperatorMetadata",
                    side_effect=lambda **kwargs: mock.Mock(**kwargs),
                ),
            ):
                first = kernel_cache.get_efc_kernel_with_epilogue(
                    "test_efc",
                    epilogue_args,
                    epilogue_source="same_epilogue",
                    base_kernel=base_kernel,
                    specialization=sum_specialization,
                )
                same = kernel_cache.get_efc_kernel_with_epilogue(
                    "test_efc",
                    epilogue_args,
                    epilogue_source="same_epilogue",
                    base_kernel=base_kernel,
                    specialization=sum_specialization,
                )
                different = kernel_cache.get_efc_kernel_with_epilogue(
                    "test_efc",
                    epilogue_args,
                    epilogue_source="same_epilogue",
                    base_kernel=base_kernel,
                    specialization=max_specialization,
                )
        finally:
            kernel_cache.clear_cache()

        self.assertIs(first, same)
        self.assertIsNot(first, different)

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

    @unittest.skipIf(IS_FBCODE, "CUTLASS Operator API is not available in fbcode")
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

    @unittest.skipIf(IS_FBCODE, "CUTLASS Operator API is not available in fbcode")
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

    def test_atomic_reduction_fusion_hooks(self):
        from torch._inductor.codegen.cuda_combined_scheduling import (
            CUDACombinedScheduling,
        )

        scheduling = CUDACombinedScheduling.__new__(CUDACombinedScheduling)
        nvgemm = MagicMock()
        triton = MagicMock()
        scheduling._nv_universal_gemm_scheduling = nvgemm
        scheduling._triton_scheduling = triton
        node1, node2 = MagicMock(), MagicMock()

        nvgemm.has_conflicting_epilogue_reductions.return_value = True
        self.assertFalse(scheduling.can_fuse_reduction_pair(node1, node2))

        nvgemm.get_fusion_pair_priority.return_value = 1
        self.assertEqual(scheduling.get_fusion_pair_priority(node1, node2), 1)
        triton.get_fusion_pair_priority.assert_not_called()

        nvgemm.get_fusion_pair_priority.return_value = 2
        triton.get_fusion_pair_priority.return_value = 4
        self.assertEqual(scheduling.get_fusion_pair_priority(node1, node2), 6)

        nvgemm.has_nvgemm_bool_output.side_effect = (False, True)
        self.assertFalse(scheduling.can_fuse_horizontal(node1, node2))

    def test_grouped_reduction_affine_index_rejects_offset(self):
        from torch._inductor.codegen.nv_universal_gemm.epilogue_lowering import (
            _matches_affine_index,
        )

        i, j, r = sympy.symbols("i j r", integer=True)
        strides = (32, 4, 1)

        def known_equals(left, right):
            return sympy.simplify(left - right) == 0

        self.assertTrue(
            _matches_affine_index(32 * i + 4 * j + r, (i, j, r), strides, known_equals)
        )
        self.assertFalse(
            _matches_affine_index(
                32 * i + 4 * j + r + 1, (i, j, r), strides, known_equals
            )
        )
        self.assertTrue(
            _matches_affine_index(32 * i + 4 * j + r, (), strides, known_equals)
        )
        self.assertFalse(
            _matches_affine_index(32 * i + 4 * j + r + 1, (), strides, known_equals)
        )

    def test_reduction_source_rejects_different_load_indices(self):
        from torch._inductor.kernel.loop_ir_cutedsl_codegen import LoopIRCuteDSLCodegen
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRExpression as Expr,
        )

        index = sympy.Symbol("index", integer=True)
        source = Expr(
            "mul",
            (
                Expr("load", ("gemm", index, None)),
                Expr("load", ("gemm", index + 1, None)),
            ),
        )
        with self.assertRaisesRegex(NotImplementedError, "one logical element"):
            LoopIRCuteDSLCodegen.source_from_expression("gemm", source, "source")

    def test_epilogue_rejects_remapped_same_shape_load(self):
        from torch._inductor.kernel.loop_ir_cutedsl_codegen import LoopIRCuteDSLCodegen
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRAnalysis,
            GemmEpilogueIRExpression as Expr,
            GemmEpilogueIRStore,
        )

        i, j = sympy.symbols("i j", integer=True)
        buffers = {name: MagicMock() for name in ("gemm", "other", "out")}
        for buffer in buffers.values():
            buffer.get_size.return_value = (4, 4)
            buffer.get_stride.return_value = (4, 1)
            buffer.get_layout.return_value.offset = 0
        graph = MagicMock(graph_inputs={})
        graph.get_buffer.side_effect = buffers.get
        graph.sizevars.statically_known_equals.side_effect = lambda a, b: a == b
        analysis = GemmEpilogueIRAnalysis(
            {
                "out": GemmEpilogueIRStore(
                    4 * i + j,
                    Expr("load", ("other", i + 4 * j, None)),
                )
            },
            index_vars={"out": (i, j)},
        )

        with (
            V.set_graph_handler(graph),
            self.assertRaisesRegex(NotImplementedError, "remapped tensor loads"),
        ):
            LoopIRCuteDSLCodegen("gemm", OrderedSet())._validate_load_indices(
                analysis, "out"
            )

    def test_output_scale_rejects_value_changing_cast(self):
        from torch._inductor.kernel.loop_ir_cutedsl_codegen import LoopIRCuteDSLCodegen
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRExpression as Expr,
        )

        scale = MagicMock()
        scale.get_dtype.return_value = torch.float32
        scale.get_size.return_value = (1,)
        graph = MagicMock(
            name_to_buffer={},
            graph_inputs={"scale": scale},
        )
        graph.sizevars.statically_known_equals.side_effect = lambda left, right: (
            left == right
        )
        codegen = LoopIRCuteDSLCodegen("gemm", OrderedSet(("gemm",)))
        accumulator = Expr("load", ("gemm", 0, None))
        scale_load = Expr("load", ("scale", 0, None))
        with V.set_graph_handler(graph):
            self.assertEqual(
                codegen._output_scale_name(Expr("mul", (accumulator, scale_load))),
                "scale",
            )
            self.assertEqual(
                codegen._output_scale_name(
                    Expr(
                        "mul",
                        (accumulator, Expr("to_dtype", (scale_load, torch.float32))),
                    )
                ),
                "scale",
            )
            self.assertIsNone(
                codegen._output_scale_name(
                    Expr(
                        "mul",
                        (accumulator, Expr("to_dtype", (scale_load, torch.int32))),
                    )
                )
            )

    def test_grouped_reduction_conversion_contract(self):
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRExpression as Expr,
            GemmEpilogueIRStore,
            grouped_reduction_pattern_ir,
        )

        load = Expr("load", ("gemm", 0, None))

        def classify(value):
            square = Expr("mul", (value, value))
            reduction = Expr("reduction", (torch.float32, torch.float32, "sum", square))
            return grouped_reduction_pattern_ir(
                GemmEpilogueIRStore(0, reduction), "gemm", 4, torch.bfloat16
            )

        fp32 = Expr("to_dtype", (load, torch.float32))
        result = classify(fp32)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "sum")
        self.assertEqual(result[1].op, "mul")
        fp16_then_fp32 = Expr(
            "to_dtype", (Expr("to_dtype", (load, torch.float16)), torch.float32)
        )
        self.assertIsNone(classify(fp16_then_fp32))
        bitcast = Expr("to_dtype_bitcast", (load, torch.bfloat16, torch.bfloat16))
        self.assertIsNone(classify(Expr("to_dtype", (bitcast, torch.float32))))

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
        self.assertTrue(capabilities.supports("max"))
        self.assertFalse(capabilities.supports("unsupported"))

        from torch._inductor.codegen.nv_universal_gemm import GemmVariant

        plan = GemmReductionPlan(
            reduction_output=None,
            group=4,
            axis=1,
            reduction_type="sum",
            source_fn="def source(value):\n    return value",
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
        self.assertFalse(GemmVariant.SCALED_GEMM.supports_reduction(wide_consumer))
        unsupported = dataclasses.replace(plan, reduction_type="unsupported")
        self.assertFalse(GemmVariant.GEMM.supports_reduction(unsupported))
        self.assertTrue(GemmVariant.SCALED_GEMM.supports_reduction(plan))
        generated_source = dataclasses.replace(
            plan, source_fn="def source(value):\n    return value * value"
        )
        self.assertTrue(GemmVariant.GEMM.supports_reduction(generated_source))
        self.assertTrue(GemmVariant.SCALED_GEMM.supports_reduction(generated_source))
        secondary = dataclasses.replace(plan, secondary_feed_output="secondary")
        self.assertFalse(GemmVariant.GEMM.supports_reduction(secondary))
        self.assertFalse(GemmVariant.SCALED_GEMM.supports_reduction(secondary))
        custom_secondary = dataclasses.replace(
            secondary, secondary_consumer_fn="generated_consumer"
        )
        self.assertTrue(GemmVariant.GEMM.supports_reduction(custom_secondary))
        self.assertTrue(
            GemmVariant.GEMM.supports_reduction(
                dataclasses.replace(custom_secondary, feeds_main=True)
            )
        )
        self.assertFalse(GemmVariant.SCALED_GEMM.supports_reduction(custom_secondary))
        normalized_secondary = dataclasses.replace(
            secondary,
            axis=0,
            feeds_main=True,
            secondary_consumer_fn="generated_consumer",
        )
        self.assertTrue(GemmVariant.GEMM.supports_reduction(normalized_secondary))
        self.assertTrue(
            GemmVariant.GEMM.supports_reduction(
                dataclasses.replace(normalized_secondary, axis=1)
            )
        )
        self.assertFalse(GemmVariant.SCALED_GEMM.supports_reduction(unsupported))
        self.assertFalse(GemmVariant.GROUPED_GEMM.supports_reduction(plan))

    def test_grouped_reduction_ir_preserves_primitive_and_source(self):
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRExpression as Expr,
            GemmEpilogueIRStore,
            grouped_reduction_pattern_ir,
        )

        load = Expr("load", ("gemm", 0, None))
        reduction = Expr("reduction", (torch.float32, torch.float32, "sum", load))
        pattern = grouped_reduction_pattern_ir(
            GemmEpilogueIRStore(0, reduction), "gemm", 4, torch.float32
        )
        self.assertIsNotNone(pattern)
        reduction_type, source = pattern
        self.assertEqual(reduction_type, "sum")
        self.assertIs(source, load)

    def test_loop_ir_epilogue_region_preserves_multiple_reductions(self):
        import sympy

        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            _GemmEpilogueIRHandler,
            GemmEpilogueIRAnalysis,
        )
        from torch._inductor.virtualized import V

        index = sympy.Symbol("i", integer=True, nonnegative=True)
        handler = _GemmEpilogueIRHandler()
        with V.set_ops_handler(handler):
            value = V.ops.load("gemm", index)
            reductions = [
                V.ops.reduction(torch.float32, torch.float32, kind, value)
                for kind in ("sum", "max", "prod")
            ]
            V.ops.store(
                "out", index, V.ops.add(reductions[0], V.ops.add(*reductions[1:]))
            )

        region = GemmEpilogueIRAnalysis(handler.stores).reduction_region(
            "out", "gemm", 16, torch.float32
        )
        self.assertIsNotNone(region)
        self.assertEqual(
            tuple(reduction.reduction_type for reduction in region.reductions),
            ("sum", "max", "prod"),
        )

    def test_reduction_consumer_receives_finalized_value(self):
        import dataclasses

        from torch._inductor.kernel import gemm_epilogue_codegen
        from torch._inductor.kernel.gemm_epilogue import (
            GEMM_REDUCTION_IDENTITY_SOURCE,
            GemmReductionArguments,
        )
        from torch._inductor.kernel.gemm_epilogue_codegen import (
            MaterializedTensorSSAReduction,
        )

        reduction = MaterializedTensorSSAReduction(
            "reduce",
            "init",
            lambda x, y: x + y,
            lambda value: value,
            lambda value, group: ("builtin", value, group),
        )
        callbacks = {
            GEMM_REDUCTION_IDENTITY_SOURCE: lambda value: value,
            "consumer": lambda accumulator, primary, secondary: (
                accumulator,
                primary,
                secondary,
            ),
            "finalizer": lambda value, group: ("generated", value, group),
            "consumer_finalizer": lambda value, group: ("consumer", value, group),
        }
        with (
            mock.patch.object(
                gemm_epilogue_codegen,
                "materialize_tensorssa_reduction",
                return_value=reduction,
            ),
            mock.patch.object(
                gemm_epilogue_codegen,
                "materialize_epilogue_function",
                side_effect=lambda source, _cute: callbacks[source],
            ),
        ):
            args = GemmReductionArguments(group=4, consumer_fn="consumer")
            compile_config = gemm_epilogue_codegen.GemmReductionCompileConfig.from_args(
                args, object()
            )
            self.assertEqual(
                compile_config.consumer("acc", "raw", "secondary"),
                ("acc", "raw", "secondary"),
            )

            args = dataclasses.replace(args, finalizer_fn="finalizer")
            compile_config = gemm_epilogue_codegen.GemmReductionCompileConfig.from_args(
                args, object()
            )
            self.assertEqual(
                compile_config.reduction.finalize("raw", 4),
                ("generated", "raw", 4),
            )
            self.assertEqual(
                compile_config.consumer("acc", "raw", "secondary"),
                ("acc", "raw", "secondary"),
            )

            args = dataclasses.replace(args, consumer_finalizer_fn="consumer_finalizer")
            compile_config = gemm_epilogue_codegen.GemmReductionCompileConfig.from_args(
                args, object()
            )
            self.assertEqual(
                compile_config.reduction.finalize("raw", 4),
                ("generated", "raw", 4),
            )
            self.assertEqual(
                compile_config.consumer("acc", "raw", "secondary"),
                ("acc", ("consumer", "raw", 4), "secondary"),
            )

            args = dataclasses.replace(
                args,
                consumer_finalizer_fn=None,
                secondary_consumer_fn="consumer",
            )
            compile_config = gemm_epilogue_codegen.GemmReductionCompileConfig.from_args(
                args, object()
            )
            self.assertEqual(
                compile_config.consumer("acc", "raw", "secondary"),
                ("acc", "raw", "secondary"),
            )
            self.assertEqual(
                compile_config.secondary_consumer("acc", "raw", "secondary"),
                ("acc", "raw", "secondary"),
            )

    def test_feed_plan_preserves_reduction_finalizer(self):
        import dataclasses

        from torch._inductor.codegen.nv_universal_gemm.epilogue_lowering import (
            NVGemmEpilogueCapture,
            NVGemmEpilogueLowering,
            NVGemmFeedPlan,
            NVGemmReductionPartition,
            NVGemmReductionRegion,
        )
        from torch._inductor.kernel.gemm_epilogue import (
            GEMM_REDUCTION_IDENTITY_SOURCE,
            GemmReductionConfig,
            GemmReductionPlan,
        )

        finalizer = "def finalize(value, group):\n    return value / group"
        config = GemmReductionConfig(
            output_name="mean",
            group=4,
            axis=1,
            reduction_type="sum",
            source_fn=GEMM_REDUCTION_IDENTITY_SOURCE,
            finalizer_fn=finalizer,
        )
        feed_plan = NVGemmFeedPlan(
            plan=GemmReductionPlan.from_config(
                dataclasses.replace(config, output_name="main", finalizer_fn=None),
                reduction_output=None,
                primary_output="out",
                feeds_main=True,
            )
        )
        gemm = mock.Mock()
        gemm.get_name.return_value = "out"
        plan = NVGemmEpilogueLowering._static_reduction_plan(
            NVGemmEpilogueCapture(gemm=gemm, nodes=(), analysis=None),
            NVGemmReductionPartition((NVGemmReductionRegion(config=config, nodes=()),)),
            feed_plan,
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.reduction_output, "mean")
        self.assertEqual(plan.finalizer_fn, finalizer)

    @unittest.skipUnless(
        ensure_nv_universal_gemm_available(), "Requires cutlass.operators"
    )
    def test_dense_reduction_uses_generated_callback_contract(self):
        import dataclasses

        import cutlass.cute as cute

        from torch._inductor.codegen.nv_universal_gemm.epilogue_capabilities import (
            BLOCK_SCALED_GEMM_REDUCTION_CAPABILITIES,
            DENSE_GEMM_REDUCTION_CAPABILITIES,
        )
        from torch._inductor.kernel.gemm_epilogue import (
            GemmReductionArguments,
            GemmReductionPlan,
        )
        from torch._inductor.kernel.gemm_epilogue_codegen import (
            GemmReductionCompileConfig,
        )

        args = GemmReductionArguments(
            group=8,
            axis=1,
            reduction_type="sum",
            source_fn="def source(value):\n    return value * value",
            feeds_main=True,
            finalizer_fn=("def finalize(value, group):\n    return value / group"),
        )
        config = GemmReductionCompileConfig.from_args(args, cute)

        self.assertEqual(
            config.constexprs()[:4],
            (8, 1, True, False),
        )
        self.assertEqual(config.reduction.source(3.0), 9.0)
        self.assertEqual(config.reduction.finalize(16.0, 8), 2.0)
        self.assertTrue(DENSE_GEMM_REDUCTION_CAPABILITIES.supports("sum"))
        secondary = dataclasses.replace(
            args,
            feeds_main=False,
            secondary_feed_output="secondary",
            secondary_consumer_fn=(
                "def consume(accumulator, primary, secondary):\n"
                "    return accumulator > 0.0"
            ),
        )
        self.assertTrue(DENSE_GEMM_REDUCTION_CAPABILITIES.supports_contract(secondary))
        self.assertFalse(
            BLOCK_SCALED_GEMM_REDUCTION_CAPABILITIES.supports_contract(secondary)
        )

        generated_args = GemmReductionArguments(
            group=4,
            axis=1,
            reduction_type=None,
            source_fn=None,
        )
        generated_config = GemmReductionCompileConfig.from_args(generated_args, cute)
        self.assertEqual(generated_config.constexprs()[:4], (4, 1, False, True))
        self.assertIsNone(generated_config.reduction.reduce_op)
        generated_plan = GemmReductionPlan(
            reduction_output="reduction",
            primary_output="output",
            group=4,
            axis=1,
            reduction_type=None,
            source_fn=None,
        )
        self.assertTrue(
            DENSE_GEMM_REDUCTION_CAPABILITIES.supports_contract(generated_plan)
        )
        self.assertTrue(
            BLOCK_SCALED_GEMM_REDUCTION_CAPABILITIES.supports_contract(generated_plan)
        )
        physical_generated_plan = GemmReductionPlan(
            reduction_output="reduction",
            primary_output="output",
            group=64,
            axis=1,
            reduction_type=None,
            source_fn=None,
            combine_fn="def combine(lhs, rhs):\n    return lhs + rhs",
            finalizer_fn="def finalize(value, group):\n    return value / group",
        )
        self.assertTrue(
            DENSE_GEMM_REDUCTION_CAPABILITIES.supports_contract(physical_generated_plan)
        )
        self.assertTrue(
            BLOCK_SCALED_GEMM_REDUCTION_CAPABILITIES.supports_contract(
                physical_generated_plan
            )
        )
        with self.assertRaisesRegex(
            RuntimeError, "specify both reduction_type and source_fn or neither"
        ):
            dataclasses.replace(generated_plan, reduction_type="sum")

    def test_grouped_reduction_rejects_ambiguous_composite(self):
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRExpression as Expr,
            GemmEpilogueIRStore,
            grouped_reduction_pattern_ir,
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
        self.assertIsNone(grouped_reduction_pattern_ir(store, "gemm", 4, torch.float32))

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
            source_fn="def source(value):\n    return abs(value)",
            primary_output="out",
            feeds_main=True,
        )
        from_plan = GemmReductionArguments.from_plan(plan, output="output")
        self.assertEqual(from_plan.output, "output")
        self.assertEqual(from_plan.group, plan.group)
        self.assertEqual(from_plan.reduction_type, plan.reduction_type)

        specialization = _local_reduce_specialization({"local_reduce": base})
        for field, value in (("group", 8), ("axis", 0), ("feeds_main", True)):
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
            feeds_main=True,
            tensor_epilogue_returns_local_reduce=False,
        )
        config = gemm_epilogue_codegen.GemmReductionCompileConfig(
            args=args,
            reduction=reduction,
            consumer="consumer",
            secondary_consumer="secondary_consumer",
        )

        self.assertIs(config.args, args)
        common = (4, 1, True, False)
        primary = (
            "reduce",
            "init",
            "combine",
            "source",
            "finalize",
            "consumer",
        )
        self.assertEqual(
            config.constexprs(),
            (*common, *primary, "secondary_consumer"),
        )

    def test_local_reduce_plan_deduplicates_outputs(self):
        from torch._inductor.kernel.gemm_epilogue import GemmReductionPlan

        plan = GemmReductionPlan(
            reduction_output="aux",
            group=4,
            axis=1,
            reduction_type="sum",
            source_fn="def source(value):\n    return value",
            primary_output="output",
            feed_output="aux",
            secondary_feed_output="output",
        )
        self.assertEqual(plan.auxiliary_outputs, ("aux",))

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
        epilogue_fused = EPILOGUE_FN_NAME in code and "EpilogueArguments" in code
        return result, code, epilogue_fused

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
        self.assertIn("CuTeDSLEpilogueArguments", code)
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
        torch.testing.assert_close(result[0], expected[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(result[1], expected[1], atol=1e-1, rtol=1e-2)
        self.assertTrue(epilogue_fused, "pointwise consumer was NOT fused")
        self.assertIn("out_ptr1", code)

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

    @parametrize(
        "case",
        (
            (1, 4, "sum"),
            (1, 16, "mean"),
            (1, 16, "prod"),
            (1, 16, "amax"),
            (1, 16, "amin"),
            (1, 16, "square_mean"),
            (1, 64, "abs_amax"),
            (0, 2, "sum"),
            (0, 16, "mean"),
            (0, 16, "prod"),
            (0, 16, "amax"),
            (0, 16, "amin"),
            (0, 16, "square_mean"),
            (0, 64, "abs_amax"),
        ),
        name_fn=lambda case: f"axis_{case[0]}_group_{case[1]}_{case[2]}",
    )
    def test_bf16_grouped_reduce_epilogue_fusion(self, case):
        axis, group, reduction = case
        m, n, k = 128, 128, 64
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = (
                result.float().view(-1, group, n)
                if axis == 0
                else result.float().view(m, -1, group)
            )
            source, _, reduce_op = reduction.rpartition("_")
            if source == "square":
                grouped = grouped.square()
            elif source == "abs":
                grouped = grouped.abs()
            reduced = getattr(grouped, reduce_op)(1 if axis == 0 else -1)
            return torch.relu(result), reduced

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertIn("VendoredDenseGemmEFCOperator", code)
        self.assertIn(f"axis={axis}", code)
        self.assertIn(f"group={group}", code)
        self.assertIn("reduction_type=None", code)
        self.assertIn("source_fn=None", code)
        self.assertNotIn("_LOCAL_REDUCE_SOURCE_FN_SRC", code)
        if axis == 0 or group > 32:
            self.assertIn("_LOCAL_REDUCE_COMBINE_FN_SRC", code)

    def test_bf16_grouped_m_reduce_finalizes_after_cross_warp_combine(self):
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
        self.assertIn("_LOCAL_REDUCE_COMBINE_FN_SRC", code)
        self.assertIn("_LOCAL_REDUCE_FINALIZER_FN_SRC", code)
        self.assertIn("group=64", code)

    def test_bf16_grouped_n_reduce_post_op_feeds_main(self):
        m, n, k, group = 128, 128, 64, 4
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            mean = grouped.sum(-1, keepdim=True) / group
            return (grouped - mean).reshape(m, n)

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertIn("feeds_main=True", code)
        self.assertNotIn("_LOCAL_REDUCE_CONSUMER_FINALIZER_FN_SRC", code)

    def test_bf16_grouped_n_reduce_raw_feed_and_finalized_output(self):
        m, n, k, group = 128, 128, 64, 4
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            sums = grouped.sum(-1, keepdim=True)
            means = sums / group
            return (grouped - sums).reshape(m, n), means

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertIn("feeds_main=True", code)
        self.assertNotIn("_LOCAL_REDUCE_CONSUMER_FINALIZER_FN_SRC", code)
        self.assertIn("_LOCAL_REDUCE_FINALIZER_FN_SRC", code)

    def test_bf16_grouped_m_mean_feeds_main_before_output_finalizer(self):
        m, n, k, group = 128, 64, 64, 64
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(-1, group, n)
            means = grouped.square().mean(1, keepdim=True)
            roots = means.sqrt()
            centered = result.float() - means.expand(-1, group, -1).reshape(m, n)
            return centered, roots

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertIn("_LOCAL_REDUCE_CONSUMER_FINALIZER_FN_SRC", code)
        self.assertIn("_LOCAL_REDUCE_FINALIZER_FN_SRC", code)

    def test_bf16_grouped_n_mean_feeds_main_before_output_finalizer(self):
        m, n, k, group = 128, 128, 64, 4
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            means = grouped.mean(-1, keepdim=True)
            roots = means.sqrt()
            return (grouped - means).reshape(m, n), roots

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertIn("_LOCAL_REDUCE_CONSUMER_FINALIZER_FN_SRC", code)
        self.assertIn("_LOCAL_REDUCE_FINALIZER_FN_SRC", code)

    def test_bf16_grouped_n_composite_reduction_fusion(self):
        m, n, k, group = 128, 64, 64, 4
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            scale = grouped.sum(-1) + grouped.amax(-1)
            return (
                torch.relu(result),
                (grouped / scale.unsqueeze(-1)).view(m, n),
                scale,
            )

        result, code, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=2e-2, rtol=2e-2)
        self.assertIn("cute.ReductionOp.ADD", code)
        self.assertIn("cute.ReductionOp.MAX", code)
        self.assertIn("reduction_type=None", code)
        self.assertIn("source_fn=None", code)

    def test_bf16_grouped_n_distinct_reduction_consumers(self):
        m, n, k, group = 128, 64, 64, 4
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            result = a @ b
            grouped = result.float().view(m, -1, group)
            sums = grouped.sum(-1, keepdim=True)
            maxima = grouped.amax(-1, keepdim=True)
            return (grouped - sums).reshape(m, n), (grouped - maxima).reshape(m, n)

        result, _, _ = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=2e-2, rtol=2e-2)

    def test_scaled_mm_grouped_n_composite_reduction_fusion(self):
        m, n, k, group = 128, 128, 512, 4
        a, b, scale_a, scale_b = _make_nvfp4_scaled_mm_inputs(m, n, k)

        def fn(a, b, scale_a, scale_b):
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            grouped = result.float().view(m, -1, group)
            magnitude = grouped.abs()
            scale = magnitude.sum(-1) + magnitude.amax(-1) + 1.0
            return (
                torch.relu(result),
                (grouped / scale.unsqueeze(-1)).view(m, n),
                scale,
            )

        result, code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
        self.assertEqual(result, fn(a, b, scale_a, scale_b), atol=2e-2, rtol=2e-2)
        self.assertIn("cute.ReductionOp.ADD", code)
        self.assertIn("cute.ReductionOp.MAX", code)
        self.assertIn("reduction_type=None", code)
        self.assertIn("source_fn=None", code)

    def test_scaled_mm_grouped_m_reduce_finalizes_after_cross_warp_combine(self):
        m, n, k, group = 128, 128, 512, 64
        a, b, scale_a, scale_b = _make_nvfp4_scaled_mm_inputs(m, n, k)

        def fn(a, b, scale_a, scale_b):
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            grouped = result.float().view(-1, group, n)
            reduced = (grouped.abs().amax(1) + 1.0).sqrt()
            return result, reduced

        result, code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
        self.assertEqual(result, fn(a, b, scale_a, scale_b), atol=1e-2, rtol=1e-2)
        self.assertIn("_LOCAL_REDUCE_COMBINE_FN_SRC", code)
        self.assertIn("_LOCAL_REDUCE_FINALIZER_FN_SRC", code)
        self.assertIn("group=64", code)

    @parametrize("operation", ("mul", "sigmoid", "gelu"))
    def test_scaled_mm_pointwise_epilogue_fusion(self, operation):
        """Unary pointwise op is fused into an NVFP4 scaled GEMM epilogue."""
        m, n, k = self.M, self.N, self.K
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

        def fn(a, b, scale_a, scale_b):
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            if operation == "sigmoid":
                return torch.sigmoid(result)
            if operation == "gelu":
                return torch.nn.functional.gelu(result)
            return result * 0.5

        result, code, epilogue_fused = self._compile_and_check(
            fn, a, b, scale_a, scale_b
        )
        self.assertIn("CuTeDSLEpilogueArguments", code)
        torch.testing.assert_close(result, fn(a, b, scale_a, scale_b), equal_nan=True)
        self.assertTrue(
            epilogue_fused, f"{operation} was NOT fused into scaled epilogue"
        )

    def test_matmul_single_store_epilogue_chain(self):
        a = torch.randn(self.M, self.K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(self.K, self.N, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            return torch.relu((a @ b).float()) + 1.0

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused)
        self.assertNotIn("out_ptr1", code)

    def test_scaled_mm_multi_store_epilogue_fusion(self):
        m, n, k = self.M, self.N, self.K
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

        def fn(a, b, scale_a, scale_b):
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            result_fp32 = result.float()
            return result, torch.relu(result_fp32), result_fp32 * 0.5

        result, code, epilogue_fused = self._compile_and_check(
            fn, a, b, scale_a, scale_b
        )
        expected = fn(a, b, scale_a, scale_b)
        self.assertEqual(result, expected)
        self.assertTrue(epilogue_fused)
        self.assertIn("out_ptr1", code)
        self.assertIn("out_ptr2", code)

    def test_scaled_mm_large_epilogue_fusion(self):
        m, n, k = self.M, self.N, self.K
        a, b, scale_a, scale_b = _make_nvfp4_scaled_mm_inputs(m, n, k)
        biases = tuple(
            torch.randn(n, device="cuda", dtype=torch.bfloat16) for _ in range(5)
        )

        def fn(a, b, scale_a, scale_b, *biases):
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            values = tuple(result + bias for bias in biases)
            return result, *values

        result, code, epilogue_fused = self._compile_and_check(
            fn, a, b, scale_a, scale_b, *biases
        )
        self.assertEqual(result, fn(a, b, scale_a, scale_b, *biases))
        self.assertTrue(epilogue_fused)
        for index in range(1, 6):
            self.assertIn(f"out_ptr{index}", code)

    @parametrize(
        "case",
        (
            ((((M, N), torch.bfloat16),), True),
            ((((N,), torch.bfloat16),), True),
            ((((1, N), torch.bfloat16),), True),
            ((((M, 1), torch.bfloat16),), True),
            ((((N,), torch.bfloat16), ((M, 1), torch.bfloat16)), True),
            ((((N,), torch.float32),), True),
        ),
        name_fn=lambda case: "_and_".join(
            f"{'x'.join(map(str, sh))}_{dtype_name(dt)}" for sh, dt in case[0]
        ),
    )
    def test_scaled_mm_broadcast_epilogue_fusion(self, case):
        bias_specs, expected_fused = case
        m, n, k = self.M, self.N, self.K
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
        biases = tuple(
            torch.randn(shape, device="cuda", dtype=dtype)
            for shape, dtype in bias_specs
        )

        def fn(a, b, scale_a, scale_b, *biases):
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            for bias in biases:
                result = result + bias
            return torch.relu(result)

        result, code, epilogue_fused = self._compile_and_check(
            fn, a, b, scale_a, scale_b, *biases
        )
        self.assertEqual(epilogue_fused, expected_fused)
        # Random packed FP4 operands can produce NaNs in the reference GEMM.
        torch.testing.assert_close(
            result, fn(a, b, scale_a, scale_b, *biases), equal_nan=True
        )

    @parametrize("bias_shape", ((1,), (1, 1)))
    def test_scaled_mm_unit_dim_broadcast_epilogue_fusion(self, bias_shape):
        m, n, k = 1, 128, self.K
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
        bias = torch.randn(bias_shape, device="cuda", dtype=torch.bfloat16)

        def fn(a, b, scale_a, scale_b, bias):
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            return torch.relu(result + bias)

        result, _, epilogue_fused = self._compile_and_check(
            fn, a, b, scale_a, scale_b, bias
        )
        self.assertTrue(epilogue_fused)
        self.assertEqual(result, fn(a, b, scale_a, scale_b, bias))

    @parametrize(
        "case",
        (
            (0, "sum", 4),
            (0, "mean", 4),
            (0, "prod", 4),
            (0, "amax", 4),
            (0, "amin", 4),
            (0, "sum", 8),
            (0, "mean", 8),
            (0, "prod", 8),
            (0, "amax", 8),
            (0, "amin", 8),
            (0, "sum", 16),
            (0, "mean", 16),
            (0, "prod", 16),
            (0, "amax", 16),
            (0, "amin", 16),
            (1, "sum", 32),
            (1, "mean", 32),
            (1, "prod", 32),
            (1, "amax", 32),
            (1, "amin", 32),
            (1, "sum", 64),
            (1, "mean", 64),
            (1, "prod", 64),
            (1, "amax", 64),
            (1, "amin", 64),
        ),
        name_fn=lambda case: f"axis_{case[0]}_{case[1]}_group_{case[2]}",
    )
    def test_scaled_mm_grouped_reduce_fusion(self, case):
        axis, reduction, group = case
        m, n, k = 128, 256 if group > 32 else 128, 512
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

        def fn(a, b, scale_a, scale_b):
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            grouped = (
                result.float().view(-1, group, n)
                if axis == 0
                else result.float().view(m, -1, group)
            )
            dim = 1 if axis == 0 else -1
            if reduction == "sum":
                reduced = grouped.sum(dim)
            elif reduction == "mean":
                reduced = grouped.mean(dim)
            elif reduction == "prod":
                reduced = grouped.prod(dim)
            elif reduction == "amax":
                reduced = grouped.amax(dim)
            else:
                reduced = grouped.amin(dim)
            return result, reduced

        result, code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
        expected = fn(a, b, scale_a, scale_b)
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])
        self.assertIn("output=", code)

        if case == (1, "sum", 32):
            if not has_triton_reduction_ordering():
                self.skipTest("requires INNER_TREE Triton")
            with config.patch({"numerics": "strict"}):
                _, strict_code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
            self.assertNotIn("'local_reduce': GemmReductionArguments", strict_code)
            self.assertIn("ReductionOrdering.INNER_TREE", strict_code)

    def test_scaled_mm_grouped_reduce_source_fusion(self):
        m, n, k, group = 128, 128, 512, 32
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

        def fn(a, b, scale_a, scale_b):
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            grouped = result.float().view(m, -1, group)
            return torch.relu(result), grouped.square().mean(-1)

        result, code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
        expected = fn(a, b, scale_a, scale_b)
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])
        self.assertIn("output=", code)
        self.assertNotIn("_LOCAL_REDUCE_SOURCE_FN_SRC", code)
        self.assertIn("reduction_type=None", code)
        self.assertIn("source_fn=None", code)
        self.assertIn(" * ", code)

    @config.patch(emulate_precision_casts=True)
    def test_scaled_mm_grouped_reduce_rejects_intermediate_fp16(self):
        m, n, k, group = 128, 128, 512, 32
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

        def fn(a, b, scale_a, scale_b):
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            grouped = result.half().float().view(m, -1, group)
            return torch.relu(result), grouped.square().mean(-1)

        result, code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
        self.assertEqual(result, fn(a, b, scale_a, scale_b))
        self.assertNotIn("'local_reduce': GemmReductionArguments", code)

    def test_scaled_mm_grouped_reduce_feeds_main(self):
        m, n, k, group = 128, 128, 512, 4
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

        def fn(a, b, scale_a, scale_b):
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            grouped = result.float().view(-1, group, n)
            mean = (grouped.sum(1, keepdim=True) / group).expand(-1, group, -1)
            return result.float() - mean.reshape(m, n)

        result, code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
        self.assertEqual(result, fn(a, b, scale_a, scale_b))
        self.assertIn("feeds_main=True", code)
        self.assertNotIn("_LOCAL_REDUCE_CONSUMER_FINALIZER_FN_SRC", code)

    def test_scaled_mm_grouped_reduce_raw_feed_and_finalized_output(self):
        m, n, k, group = 128, 128, 512, 4
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

        def fn(a, b, scale_a, scale_b):
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            grouped = result.float().view(-1, group, n)
            sums = grouped.sum(1, keepdim=True)
            means = sums / group
            return result.float() - sums.expand(-1, group, -1).reshape(m, n), means

        result, code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
        self.assertEqual(result, fn(a, b, scale_a, scale_b))
        self.assertIn("feeds_main=True", code)
        self.assertNotIn("_LOCAL_REDUCE_CONSUMER_FINALIZER_FN_SRC", code)
        self.assertIn("_LOCAL_REDUCE_FINALIZER_FN_SRC", code)

    def test_scaled_mm_grouped_n_mean_feeds_main_before_output_finalizer(self):
        m, n, k, group = 128, 128, 512, 4
        a, b, scale_a, scale_b = _make_nvfp4_scaled_mm_inputs(m, n, k)

        def fn(a, b, scale_a, scale_b):
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            grouped = result.float().view(m, -1, group)
            means = grouped.square().mean(-1, keepdim=True)
            roots = means.sqrt()
            centered = (grouped - means).reshape(m, n)
            return centered, roots

        result, code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
        self.assertEqual(result, fn(a, b, scale_a, scale_b))
        self.assertIn("_LOCAL_REDUCE_CONSUMER_FINALIZER_FN_SRC", code)
        self.assertIn("_LOCAL_REDUCE_FINALIZER_FN_SRC", code)

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

    def test_epilogue_with_remapped_aux_input_falls_back(self):
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)
        bias = torch.randn(self.N, self.M, device="cuda", dtype=dtype)

        def fn(a, b, bias):
            return (a @ b).float() + bias.T

        result, code, epilogue_fused = self._compile_and_check(fn, a, b, bias)
        self.assertEqual(result, fn(a, b, bias), atol=1e-2, rtol=1e-2)
        self.assertFalse(epilogue_fused)
        self.assertIn("triton_poi", code)

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
