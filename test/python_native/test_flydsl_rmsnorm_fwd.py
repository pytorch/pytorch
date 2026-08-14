# Owner(s): ["module: dsl-native-ops"]

import os
import unittest
from unittest.mock import patch

import torch
import torch.backends.python_native as pn
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


EPS = 1e-5
DISPATCH_M = 8192
DISPATCH_N = 4096
DISPATCH_DTYPE = torch.float16
BACKWARD_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Covers every kernel path at every block-size tier. Three entries carry a
# reason of their own. 3 is narrower than one vector, so it has no full vector
# step and is all scalar tail. 2056 is the only entry that takes the generic
# path with more than one vector step and no tail. 8192 is the band-2 boundary.
KERNEL_PATH_NS = (
    3,
    2056,
    4096,
    8192,
    12288,
    24576,
    114688,
    4097,
    4103,
    8193,
    16385,
    32769,
    98305,
)
# (N, minimum rows M) for each band _fused_rms_norm_fwd_perf_wins admits.
BAND_MINIMUMS = ((4096, 8192), (8192, 4096), (16384, 2048))
UPPER_BOUND_N = 114688


def _unsupported_environment_reason() -> str | None:
    """Why this machine cannot exercise the override, or None if it can.

    Only environmental facts belong here. Whether the override actually
    registered on a machine that satisfies all of them is a property of the
    gate and the registry, so it is asserted by a test rather than folded in
    here -- a regression there must fail, not quietly skip this whole class.
    """
    if not TEST_CUDA or torch.version.hip is None:
        return "ROCm required"

    from torch._native import flydsl_utils as fu

    if fu.check_native_jit_disabled():
        return "native DSL overrides are disabled via TORCH_DISABLE_NATIVE_JIT"
    if not fu.runtime_available():
        return "FlyDSL runtime is not installed"
    if not fu._version_is_ok():
        return f"FlyDSL {fu.runtime_version()} is outside the supported release"

    from torch._native.ops.norm.flydsl_rmsnorm_impl import _is_supported_arch

    if not _is_supported_arch(torch.cuda.current_device()):
        return "FlyDSL RMSNorm override requires gfx950"
    return None


_UNSUPPORTED_REASON = _unsupported_environment_reason()


class TestFlyDSLRMSNormArch(TestCase):
    @parametrize(
        "arch,expected",
        (
            ("gfx950", True),
            # _resolve_rocm_arch returns HSA_OVERRIDE_GFX_VERSION verbatim, so
            # the gate has to tolerate feature flags rather than compare the
            # whole string.
            ("gfx950:sramecc+", True),
            ("gfx950:sramecc+:xnack-", True),
            ("gfx942", False),
            ("gfx942:sramecc+", False),
            (None, False),
        ),
    )
    def test_arch_gate_allows_only_gfx950(self, arch, expected):
        import torch._native.ops.norm.flydsl_rmsnorm_impl as flydsl_rmsnorm_impl

        arch_is_supported = flydsl_rmsnorm_impl._is_supported_arch
        arch_is_supported.cache_clear()
        self.addCleanup(arch_is_supported.cache_clear)
        with patch.object(
            flydsl_rmsnorm_impl.fu, "_resolve_rocm_arch", return_value=arch
        ):
            self.assertEqual(arch_is_supported(0), expected)

    def test_arch_gate_is_resolved_once_per_process(self):
        import torch._native.ops.norm.flydsl_rmsnorm_impl as flydsl_rmsnorm_impl

        arch_is_supported = flydsl_rmsnorm_impl._is_supported_arch
        arch_is_supported.cache_clear()
        self.addCleanup(arch_is_supported.cache_clear)

        with patch.dict(os.environ, {"FLYDSL_GPU_ARCH": "gfx950"}):
            self.assertTrue(arch_is_supported(0))
        with patch.dict(os.environ, {"FLYDSL_GPU_ARCH": "gfx942"}):
            self.assertTrue(arch_is_supported(0))


class TestFlyDSLRMSNormHelpers(TestCase):
    """Helpers that touch neither flydsl nor a device.

    These run everywhere, including the machines where the kernel tests below
    are skipped for lack of a gfx950 device.
    """

    @parametrize(
        "normalized_shape,expected",
        (
            (128, 128),
            ([128], 128),
            ((128,), 128),
            # More than one normalized dimension: the kernel flattens only the
            # last one, so these decline rather than guess.
            ([2, 64], None),
            ([], None),
            # Not a sequence, and a one-element sequence that is not an int.
            (None, None),
            (["n"], None),
        ),
    )
    def test_normalized_shape_1d(self, normalized_shape, expected):
        from torch._native.ops.norm.flydsl_rmsnorm_utils import normalized_shape_1d

        self.assertEqual(normalized_shape_1d(normalized_shape), expected)

    @parametrize(
        "rows_m,n,itemsize,expected",
        (
            (2048, 114688, 4, True),
            (16383, 65536, 4, True),
            (16384, 65536, 4, False),
            (16385, 65536, 4, False),
            (1 << 31, 1, 1, False),
            (1, 1 << 31, 1, False),
        ),
    )
    def test_flydsl_buffer_span(self, rows_m, n, itemsize, expected):
        from torch._native.ops.norm.flydsl_rmsnorm_impl import _fits_int32_buffer_span

        self.assertEqual(_fits_int32_buffer_span(rows_m, n, itemsize), expected)

    def test_impl_without_weight_raises(self):
        # The predicate declines weight=None, so reaching the impl means a
        # caller bypassed it. The error has to name the missing argument
        # instead of surfacing from somewhere inside the kernel wrapper.
        from torch._native.ops.norm.flydsl_rmsnorm_impl import _fused_rms_norm_impl

        x = make_tensor((8, 128), device="cpu", dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "requires an explicit weight"):
            _fused_rms_norm_impl(x, [128], None, EPS)


@unittest.skipIf(_UNSUPPORTED_REASON is not None, str(_UNSUPPORTED_REASON))
class TestFlyDSLRMSNorm(TestCase):
    def setUp(self):
        super().setUp()
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import clear_rmsnorm_caches

        clear_rmsnorm_caches()
        torch.manual_seed(0)

    def _make_inputs(self, m, n, dtype, *, requires_grad=False):
        x = make_tensor((m, n), device="cuda", dtype=dtype, requires_grad=requires_grad)
        weight = make_tensor(
            (n,), device="cuda", dtype=dtype, requires_grad=requires_grad
        )
        return x, weight

    def _make_dispatch_inputs(self):
        return self._make_inputs(DISPATCH_M, DISPATCH_N, DISPATCH_DTYPE)

    def _assert_no_flydsl_compiles(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 0)

    def _assert_nonfinite_matches(self, actual, expected):
        self.assertEqual(torch.isnan(actual), torch.isnan(expected))
        self.assertEqual(torch.isinf(actual), torch.isinf(expected))
        finite = torch.isfinite(expected)
        self.assertEqual(actual[finite], expected[finite])

    def test_override_is_registered(self):
        self.assertIn("_fused_rms_norm", pn.get_dsl_operations("flydsl"))

    @parametrize("dtype", BACKWARD_DTYPES)
    def test_backward_through_override_matches_aten(self, dtype):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import (
            clear_rmsnorm_caches,
            rmsnorm_cache_info,
        )

        clear_rmsnorm_caches()
        x, weight = self._make_inputs(DISPATCH_M, DISPATCH_N, dtype, requires_grad=True)
        grad_out = make_tensor(x.shape, device=x.device, dtype=x.dtype)

        def grads():
            out = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
            return torch.autograd.grad(out, (x, weight), grad_out)

        with pn.flydsl.disabled():
            ref_dx, ref_dw = grads()
        self._assert_no_flydsl_compiles()

        got_dx, got_dw = grads()
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)
        self.assertEqual(got_dx, ref_dx)
        self.assertEqual(got_dw, ref_dw)

    @parametrize("eps_kind", ("negative", "nan"))
    def test_invalid_eps_falls_back_without_compiling(self, eps_kind):
        m, n = 2048, 16384
        x, weight = self._make_inputs(m, n, torch.float32)
        eps = -1.0 if eps_kind == "negative" else float("nan")

        torch.ops.aten._fused_rms_norm(x, [n], weight, eps)
        self._assert_no_flydsl_compiles()

    def test_zero_eps_dispatches(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        m, n = 2048, 16384
        x, weight = self._make_inputs(m, n, torch.float32)
        with pn.flydsl.disabled():
            ref, ref_rstd = torch.ops.aten._fused_rms_norm(x, [n], weight, 0.0)
        got, got_rstd = torch.ops.aten._fused_rms_norm(x, [n], weight, 0.0)

        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)
        self.assertEqual(got, ref)
        self.assertEqual(got_rstd, ref_rstd)

    def test_direct_forward_matches_aten_and_reuses_cache(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import (
            rmsnorm_cache_info,
            rmsnorm_fwd,
        )

        x, weight = self._make_inputs(16, 128, torch.float16)
        with pn.flydsl.disabled():
            ref_out, _ = torch.ops.aten._fused_rms_norm(x, [128], weight, EPS)

        out, _ = rmsnorm_fwd(x, [128], weight, EPS)
        self.assertEqual(out, ref_out)

        # A 3-D input with the same N/dtype/device must hit the same dynamic-M
        # specialization instead of compiling a second kernel.
        x3 = make_tensor((2, 16, 128), device="cuda", dtype=x.dtype)
        with pn.flydsl.disabled():
            ref_out3, ref_rstd3 = torch.ops.aten._fused_rms_norm(x3, [128], weight, EPS)
        out3, rstd3 = rmsnorm_fwd(x3, [128], weight, EPS)
        self.assertEqual(out3.shape, x3.shape)
        self.assertEqual(rstd3.shape, (2, 16, 1))
        self.assertEqual(out3, ref_out3)
        self.assertEqual(rstd3, ref_rstd3)

        info = rmsnorm_cache_info()["fwd"]
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 1)
        self.assertEqual(info.currsize, 1)

    @unittest.skipIf(torch.cuda.device_count() < 2, "needs two CUDA devices")
    def test_each_device_gets_its_own_specialization(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import (
            rmsnorm_cache_info,
            rmsnorm_fwd,
        )

        n = 128
        for index in (0, 1):
            device = torch.device("cuda", index)
            x = make_tensor((16, n), device=device, dtype=torch.float16)
            weight = make_tensor((n,), device=device, dtype=x.dtype)
            with pn.flydsl.disabled():
                ref, ref_rstd = torch.ops.aten._fused_rms_norm(x, [n], weight, EPS)
            got, got_rstd = rmsnorm_fwd(x, [n], weight, EPS)

            self.assertEqual(got.device, device)
            self.assertEqual(got, ref)
            self.assertEqual(got_rstd, ref_rstd)

        info = rmsnorm_cache_info()["fwd"]
        self.assertEqual(info.misses, 2)
        self.assertEqual(info.currsize, 2)

    @parametrize("mismatch", ("shape", "dtype", "device", "noncontiguous"))
    def test_weight_mismatch_is_declined(self, mismatch):
        from torch._native.ops.norm.flydsl_rmsnorm_impl import _common_supported

        n = DISPATCH_N
        x, weight = self._make_dispatch_inputs()
        self.assertTrue(_common_supported(x, n, weight))

        if mismatch == "shape":
            weight = make_tensor((n + 1,), device=x.device, dtype=x.dtype)
        elif mismatch == "dtype":
            weight = weight.to(torch.float32)
        elif mismatch == "device":
            weight = weight.cpu()
        else:
            weight = make_tensor((2 * n,), device=x.device, dtype=x.dtype)[::2]
            self.assertFalse(weight.is_contiguous())

        self.assertFalse(_common_supported(x, n, weight))

    def test_noncontiguous_weight_falls_back_to_aten(self):
        x, _ = self._make_dispatch_inputs()
        weight = make_tensor((2 * DISPATCH_N,), device=x.device, dtype=x.dtype)[::2]

        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        got = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)

        self.assertEqual(got, ref)
        self._assert_no_flydsl_compiles()

    def test_nondefault_stream_reuses_specialization(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import (
            rmsnorm_cache_info,
            rmsnorm_fwd,
        )

        x, weight = self._make_inputs(16, 128, torch.float16)
        with pn.flydsl.disabled():
            ref_out, ref_rstd = torch.ops.aten._fused_rms_norm(x, [128], weight, EPS)

        rmsnorm_fwd(x, [128], weight, EPS)
        stream = torch.cuda.Stream(device=x.device)
        with torch.cuda.stream(stream):
            out, rstd = rmsnorm_fwd(x, [128], weight, EPS)
        stream.synchronize()

        self.assertEqual(out, ref_out)
        self.assertEqual(rstd, ref_rstd)
        info = rmsnorm_cache_info()["fwd"]
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 1)
        self.assertEqual(info.currsize, 1)

    @parametrize("dtype", BACKWARD_DTYPES)
    @parametrize("n", KERNEL_PATH_NS)
    def test_numerics_across_kernel_paths(self, dtype, n):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_fwd

        rows = 8
        x, weight = self._make_inputs(rows, n, dtype)
        with pn.flydsl.disabled():
            ref, ref_rstd = torch.ops.aten._fused_rms_norm(x, [n], weight, EPS)
        got, got_rstd = rmsnorm_fwd(x, [n], weight, EPS)

        self.assertEqual(got, ref)
        self.assertEqual(got_rstd.dtype, torch.float32)
        self.assertEqual(got_rstd.shape, (rows, 1))
        self.assertEqual(got_rstd, ref_rstd)

    @parametrize("dtype", BACKWARD_DTYPES)
    def test_nonfinite_input_matches_aten(self, dtype):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_fwd

        x, weight = self._make_inputs(2, 128, dtype)
        x[0, 0] = float("nan")
        x[1, 0] = float("inf")
        with pn.flydsl.disabled():
            ref, ref_rstd = torch.ops.aten._fused_rms_norm(x, [128], weight, EPS)
        got, got_rstd = rmsnorm_fwd(x, [128], weight, EPS)

        self._assert_nonfinite_matches(got, ref)
        self._assert_nonfinite_matches(got_rstd, ref_rstd)

    def test_public_rms_norm_dispatches_to_flydsl(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        x, weight = self._make_dispatch_inputs()
        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        got = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)

        self.assertEqual(got, ref)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)

    def test_cold_cuda_graph_capture_dispatches_and_replays(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        x, weight = self._make_dispatch_inputs()
        self._assert_no_flydsl_compiles()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)

        x.neg_()
        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        got.zero_()
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(got, ref)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)

    @parametrize("n,min_rows", BAND_MINIMUMS)
    def test_band_minimum_dispatches(self, n, min_rows):
        # The numerics tests below drive the kernel directly at rows=8. M is
        # dynamic in the compiled kernel, so the shape that actually reaches
        # production -- thousands of blocks rather than one wave of them --
        # only gets compared against aten here.
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        x, weight = self._make_inputs(min_rows, n, DISPATCH_DTYPE)
        with pn.flydsl.disabled():
            ref, ref_rstd = torch.ops.aten._fused_rms_norm(x, [n], weight, EPS)
        self._assert_no_flydsl_compiles()

        got, got_rstd = torch.ops.aten._fused_rms_norm(x, [n], weight, EPS)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)
        self.assertEqual(got, ref)
        self.assertEqual(got_rstd, ref_rstd)

    @parametrize("n,min_rows", BAND_MINIMUMS)
    def test_one_row_below_band_minimum_falls_back(self, n, min_rows):
        # Pins the >= in each band. Without this, widening or narrowing a row
        # threshold by one changes which shapes dispatch and nothing goes red.
        x, weight = self._make_inputs(min_rows - 1, n, DISPATCH_DTYPE)
        torch.ops.aten._fused_rms_norm(x, [n], weight, EPS)
        self._assert_no_flydsl_compiles()

    @parametrize("n,dispatches", ((UPPER_BOUND_N, True), (UPPER_BOUND_N + 1, False)))
    def test_upper_bound_is_inclusive(self, n, dispatches):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        x, weight = self._make_inputs(2048, n, DISPATCH_DTYPE)
        with pn.flydsl.disabled():
            ref, ref_rstd = torch.ops.aten._fused_rms_norm(x, [n], weight, EPS)
        self._assert_no_flydsl_compiles()

        got, got_rstd = torch.ops.aten._fused_rms_norm(x, [n], weight, EPS)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1 if dispatches else 0)
        self.assertEqual(got, ref)
        self.assertEqual(got_rstd, ref_rstd)

    def test_nd_input_dispatches_and_matches_aten(self):
        # rmsnorm_fwd flattens to (M, N) and rebuilds a different stat_shape for
        # ndim != 2, and (batch, seq, hidden) is what a model actually passes --
        # but every other dispatching test here, and every OpInfo sample, is 2-D.
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        shape = (4, 2048, DISPATCH_N)
        x = make_tensor(shape, device="cuda", dtype=DISPATCH_DTYPE)
        weight = make_tensor((DISPATCH_N,), device="cuda", dtype=DISPATCH_DTYPE)

        with pn.flydsl.disabled():
            ref, ref_rstd = torch.ops.aten._fused_rms_norm(x, [DISPATCH_N], weight, EPS)
        self._assert_no_flydsl_compiles()

        got, got_rstd = torch.ops.aten._fused_rms_norm(x, [DISPATCH_N], weight, EPS)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)
        self.assertEqual(got.shape, shape)
        self.assertEqual(got_rstd.shape, (*shape[:-1], 1))
        self.assertEqual(got, ref)
        self.assertEqual(got_rstd, ref_rstd)

    def test_nn_rmsnorm_default_eps_dispatches(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        module = torch.nn.RMSNorm(DISPATCH_N, device="cuda", dtype=DISPATCH_DTYPE)
        self.assertIsNone(module.eps)
        with torch.no_grad():
            module.weight.copy_(
                make_tensor((DISPATCH_N,), device="cuda", dtype=DISPATCH_DTYPE)
            )
        x = make_tensor((DISPATCH_M, DISPATCH_N), device="cuda", dtype=DISPATCH_DTYPE)

        with pn.flydsl.disabled():
            ref = module(x)
        got = module(x)

        self.assertEqual(got, ref)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)

    def test_multi_dim_normalized_shape_raises(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_fwd

        x, weight = self._make_inputs(8, 128, torch.float16)
        with self.assertRaisesRegex(ValueError, "one normalized dimension"):
            rmsnorm_fwd(x.view(8, 8, 16), [8, 16], weight, EPS)

    def test_unsupported_dtype_raises(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_fwd

        x, weight = self._make_inputs(8, 128, torch.float64)
        with self.assertRaisesRegex(TypeError, "unsupported RMSNorm dtype"):
            rmsnorm_fwd(x, [128], weight, EPS)

    def test_dtype_tables_agree(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import FLYDSL_DTYPE_CONFIGS
        from torch._native.ops.norm.flydsl_rmsnorm_utils import SUPPORTED_DTYPES

        self.assertEqual(set(SUPPORTED_DTYPES.values()), set(FLYDSL_DTYPE_CONFIGS))

    def test_fused_aten_noncontiguous_input_falls_back_without_compiling(self):
        base = make_tensor(
            (DISPATCH_N, DISPATCH_M), device="cuda", dtype=DISPATCH_DTYPE
        )
        x = base.transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        weight = make_tensor((DISPATCH_N,), device="cuda", dtype=x.dtype)

        torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        self._assert_no_flydsl_compiles()

    @parametrize("dtype", (torch.float16, torch.float32))
    def test_misaligned_base_dispatches_and_matches_aten(self, dtype):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        storage = make_tensor(
            (DISPATCH_M * DISPATCH_N + 1,), device="cuda", dtype=dtype
        )
        weight_storage = make_tensor((DISPATCH_N + 1,), device="cuda", dtype=dtype)
        x = storage[1:].view(DISPATCH_M, DISPATCH_N)
        weight = weight_storage[1:]
        self.assertTrue(x.is_contiguous())
        self.assertNotEqual(x.data_ptr() % 16, 0)
        self.assertNotEqual(weight.data_ptr() % 16, 0)

        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        got = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)

        self.assertEqual(got, ref)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)

    @parametrize("cow_arg", ("input", "weight"))
    def test_cow_inputs_dispatch_and_remain_cow(self, cow_arg):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        x, weight = self._make_dispatch_inputs()
        with pn.flydsl.disabled():
            ref, ref_rstd = torch.ops.aten._fused_rms_norm(x, [DISPATCH_N], weight, EPS)

        if cow_arg == "input":
            x = x._lazy_clone()
            cow = x
        else:
            weight = weight._lazy_clone()
            cow = weight
        self.assertTrue(torch._C._is_cow_tensor(cow))
        data_ptr = cow.const_data_ptr()

        got, got_rstd = torch.ops.aten._fused_rms_norm(x, [DISPATCH_N], weight, EPS)

        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)
        self.assertEqual(got, ref)
        self.assertEqual(got_rstd, ref_rstd)
        self.assertTrue(torch._C._is_cow_tensor(cow))
        self.assertEqual(cow.const_data_ptr(), data_ptr)

    def test_user_disable_falls_back_and_restores(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        x, weight = self._make_dispatch_inputs()

        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
            self._assert_no_flydsl_compiles()

        got = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        self.assertEqual(got, ref)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)


instantiate_parametrized_tests(TestFlyDSLRMSNormArch)
instantiate_parametrized_tests(TestFlyDSLRMSNormHelpers)
instantiate_parametrized_tests(TestFlyDSLRMSNorm)


if __name__ == "__main__":
    run_tests()
