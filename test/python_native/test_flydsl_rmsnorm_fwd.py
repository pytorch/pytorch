# Owner(s): ["module: dsl-native-ops"]

import unittest

import torch
import torch.backends.python_native as pn
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase


EPS = 1e-5
DISPATCH_M = 8192
DISPATCH_N = 4096
DISPATCH_DTYPE = torch.float16


def _flydsl_rmsnorm_registered() -> bool:
    try:
        operations = set(pn.get_dsl_operations("flydsl"))
        return "_fused_rms_norm" in operations
    except Exception:
        return False


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    return {
        torch.float32: (1e-4, 1e-3),
        torch.float16: (3e-2, 3e-2),
        torch.bfloat16: (1e-1, 2e-1),
    }[dtype]


def _assert_close(test_case, actual, expected, dtype):
    rtol, atol = _tolerance(dtype)
    test_case.assertEqual(actual.shape, expected.shape)
    test_case.assertEqual(actual, expected, rtol=rtol, atol=atol)


@unittest.skipUnless(TEST_CUDA and torch.version.hip is not None, "ROCm required")
@unittest.skipUnless(
    _flydsl_rmsnorm_registered(), "FlyDSL RMSNorm overrides are not registered"
)
class TestFlyDSLRMSNorm(TestCase):
    def setUp(self):
        super().setUp()
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import clear_rmsnorm_caches

        clear_rmsnorm_caches()
        torch.manual_seed(0)

    def _make_inputs(self, m, n, dtype, *, requires_grad=False):
        x = torch.randn((m, n), device="cuda", dtype=dtype, requires_grad=requires_grad)
        weight = torch.randn(
            (n,), device="cuda", dtype=dtype, requires_grad=requires_grad
        )
        return x, weight

    def _make_dispatch_inputs(self):
        return self._make_inputs(DISPATCH_M, DISPATCH_N, DISPATCH_DTYPE)

    def _assert_no_flydsl_compiles(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 0)

    def test_backward_through_override_matches_aten(self):
        # The override sits at the CUDA key, below Autograd, so ATen's
        # _fused_rms_norm_backward_cuda consumes the rstd this kernel writes:
        # fp32, shaped (*batch, 1). A drift in that contract shows up as wrong
        # gradients rather than an error, and OpInfo's test_backward only runs
        # fp32, so fp16/bf16 -- where the fp32 rstd meets a lower-precision
        # input -- would otherwise go unchecked.
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import (
            clear_rmsnorm_caches,
            rmsnorm_cache_info,
        )

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                clear_rmsnorm_caches()
                x, weight = self._make_inputs(
                    DISPATCH_M, DISPATCH_N, dtype, requires_grad=True
                )
                grad_out = torch.randn_like(x)

                def grads():
                    out = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
                    return torch.autograd.grad(out, (x, weight), grad_out)

                with pn.flydsl.disabled():
                    ref_dx, ref_dw = grads()
                self._assert_no_flydsl_compiles()

                got_dx, got_dw = grads()
                self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)
                _assert_close(self, got_dx, ref_dx, dtype)
                _assert_close(self, got_dw, ref_dw, dtype)

    def test_direct_forward_matches_aten_and_reuses_cache(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import (
            rmsnorm_cache_info,
            rmsnorm_fwd,
        )

        x, weight = self._make_inputs(16, 128, torch.float16)
        with pn.flydsl.disabled():
            ref_out, _ = torch.ops.aten._fused_rms_norm(x, [128], weight, EPS)

        out, rstd = rmsnorm_fwd(x, [128], weight, EPS)
        self.assertEqual(rstd.dtype, torch.float32)
        self.assertEqual(rstd.shape, (16, 1))
        self.assertEqual(rstd.device, x.device)
        _assert_close(self, out, ref_out, x.dtype)

        # A 3-D input with the same N/dtype/device must hit the same dynamic-M
        # specialization instead of compiling a second kernel.
        x3 = torch.randn((2, 16, 128), device="cuda", dtype=x.dtype)
        out3, _ = rmsnorm_fwd(x3, [128], weight, EPS)
        self.assertEqual(out3.shape, x3.shape)

        info = rmsnorm_cache_info()["fwd"]
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 1)
        self.assertEqual(info.currsize, 1)

    def test_numerics_across_kernel_paths_and_block_sizes(self):
        # OpInfo covers the shapes the dispatcher accepts in the common case;
        # this walks the kernel's internal branches instead, which OpInfo has
        # no way to target. Each N is chosen for a specific one:
        #
        #   4096   fast path, 256 threads
        #   12288  fast path, 512 threads (_forward_block_threads boundary)
        #   24576  fast path, 1024 threads (second boundary)
        #   114688 fast path, upper end of the dispatcher's N range
        #
        # The rest are odd, so N % vec_width != 0 for both vector widths (8 for
        # fp16/bf16, 4 for fp32) and they take the generic path's scalar tail.
        # 4097/8193/16385/32769 are the shapes the PR reports the largest wins
        # on, so the tail is the branch those numbers actually come from. Those
        # all leave exactly one element in the tail; 4103 leaves seven (three
        # for fp32) so the tail's own bounds check is exercised too.
        #
        # Rows are kept small and rmsnorm_fwd is called directly: this is about
        # the reduction being right, not about dispatch.
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_fwd

        rows = 8
        fast_path = (4096, 12288, 24576, 114688)
        scalar_tail = (4097, 4103, 8193, 16385, 32769, 98305)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for n in fast_path + scalar_tail:
                with self.subTest(dtype=dtype, n=n):
                    x, weight = self._make_inputs(rows, n, dtype)
                    with pn.flydsl.disabled():
                        ref, ref_rstd = torch.ops.aten._fused_rms_norm(
                            x, [n], weight, EPS
                        )
                    got, got_rstd = rmsnorm_fwd(x, [n], weight, EPS)

                    _assert_close(self, got, ref, dtype)
                    self.assertEqual(got_rstd.dtype, torch.float32)
                    self.assertEqual(got_rstd.shape, (rows, 1))
                    # rstd is the reduction result itself, so it is compared at
                    # fp32 tolerance regardless of the input dtype.
                    self.assertEqual(got_rstd, ref_rstd, rtol=1e-5, atol=1e-6)

    def test_public_rms_norm_dispatches_to_flydsl(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        x, weight = self._make_dispatch_inputs()
        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        got = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)

        _assert_close(self, got, ref, x.dtype)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)

    def test_runtime_eps_dispatches_and_reuses_cache(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        x, weight = self._make_dispatch_inputs()
        for eps in (EPS, 1e-6):
            with pn.flydsl.disabled():
                ref = torch.rms_norm(x, (DISPATCH_N,), weight, eps)
            got = torch.rms_norm(x, (DISPATCH_N,), weight, eps)
            _assert_close(self, got, ref, x.dtype)

        info = rmsnorm_cache_info()["fwd"]
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 1)
        self.assertEqual(info.currsize, 1)

    def test_n_above_upper_bound_falls_back_without_compiling(self):
        # 114688 is the largest N the perf table accepts; the row cache spills
        # to scratch beyond it and the kernel loses to ATen.
        n = 131072
        x = torch.randn((2048, n), device="cuda", dtype=DISPATCH_DTYPE)
        weight = torch.randn((n,), device="cuda", dtype=DISPATCH_DTYPE)

        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (n,), weight, EPS)
        got = torch.rms_norm(x, (n,), weight, EPS)

        self.assertEqual(got, ref)
        self._assert_no_flydsl_compiles()

    def test_fused_aten_noncontiguous_input_falls_back_without_compiling(self):
        base = torch.randn(
            (DISPATCH_N, DISPATCH_M), device="cuda", dtype=DISPATCH_DTYPE
        )
        x = base.transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        weight = torch.randn((DISPATCH_N,), device="cuda", dtype=x.dtype)

        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        got = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)

        self.assertEqual(got, ref)
        self._assert_no_flydsl_compiles()

    def test_misaligned_base_dispatches_and_matches_aten(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import (
            clear_rmsnorm_caches,
            rmsnorm_cache_info,
        )

        # Both dtypes issue 128-bit copies (fp32 4x32, fp16 8x16), so both hit
        # the kernel with a base address that is not 16-byte aligned.
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                clear_rmsnorm_caches()
                storage = torch.randn(
                    (DISPATCH_M * DISPATCH_N + 1,), device="cuda", dtype=dtype
                )
                weight_storage = torch.randn(
                    (DISPATCH_N + 1,), device="cuda", dtype=dtype
                )
                x = storage[1:].view(DISPATCH_M, DISPATCH_N)
                weight = weight_storage[1:]
                self.assertTrue(x.is_contiguous())
                self.assertNotEqual(x.data_ptr() % 16, 0)
                self.assertNotEqual(weight.data_ptr() % 16, 0)

                with pn.flydsl.disabled():
                    ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
                got = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)

                _assert_close(self, got, ref, dtype)
                self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)

    def test_user_disable_falls_back_and_restores(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        # A shape the dispatcher would otherwise take, so being disabled is the
        # only reason nothing compiles inside the block.
        x, weight = self._make_dispatch_inputs()

        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
            self._assert_no_flydsl_compiles()

        # Leaving the block must re-enable the override, so this call is the
        # FlyDSL kernel and the comparison against ref is aten vs FlyDSL.
        got = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        _assert_close(self, got, ref, x.dtype)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)


if __name__ == "__main__":
    run_tests()
