# Owner(s): ["module: dsl-native-ops"]

import unittest

import torch
import torch.backends.python_native as pn
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


EPS = 1e-5
# This shape is below the FWD gate but wins for BWD, which exercises that the
# two operator predicates dispatch independently.
BWD_DISPATCH_M = 2048
BWD_DISPATCH_N = 8192
BWD_DISPATCH_DTYPE = torch.float16

DIRECT_CASES = (
    (128, 4096, torch.float16),
    (128, 8192, torch.bfloat16),
    (128, 4096, torch.float32),
    # Non-aligned N exercises the generic vector body and scalar tail in
    # K1/K2/K3. M=65 also crosses the 64-way dweight split boundary.
    (65, 4197, torch.float16),
    (65, 4197, torch.bfloat16),
    (65, 4197, torch.float32),
    # Tail-only cases cover N smaller than one 128-bit vector.
    (65, 7, torch.float16),
    (65, 3, torch.float32),
)

# dtype, inclusive N range, first enabled M, and previous M kept on ATen.
BWD_PERF_RANGES = (
    (torch.float16, 16, 63, 8192, 4096),
    (torch.float16, 1024, 2047, 65536, 32768),
    (torch.float16, 2048, 4095, 32768, 16384),
    (torch.float16, 4096, 8191, 8192, 4096),
    (torch.float16, 8192, 16383, 2048, 1024),
    (torch.float16, 16384, 32767, 512, 256),
    (torch.float16, 32768, 65536, 16, 8),
    (torch.bfloat16, 16, 63, 8192, 4096),
    (torch.bfloat16, 1024, 2047, 65536, 32768),
    (torch.bfloat16, 2048, 4095, 32768, 16384),
    (torch.bfloat16, 4096, 8191, 8192, 4096),
    (torch.bfloat16, 8192, 16383, 2048, 1024),
    (torch.bfloat16, 16384, 32767, 512, 256),
    (torch.bfloat16, 32768, 65536, 16, 8),
    (torch.float32, 16, 63, 16384, 8192),
    (torch.float32, 256, 511, 65536, 32768),
    (torch.float32, 512, 2047, 16384, 8192),
    (torch.float32, 2048, 4095, 8192, 4096),
    (torch.float32, 4096, 8191, 4096, 2048),
    (torch.float32, 8192, 16383, 2048, 1024),
    (torch.float32, 16384, 32767, 64, 32),
    (torch.float32, 32768, 65536, 16, 8),
)


def _flydsl_bwd_registered() -> bool:
    try:
        return "_fused_rms_norm_backward" in pn.get_dsl_operations("flydsl")
    except Exception:
        return False


def _cache_counts() -> dict[str, int]:
    from torch._native.ops.norm.flydsl_rmsnorm_bwd import rmsnorm_bwd_cache_info

    info = rmsnorm_bwd_cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "currsize": info.currsize,
    }


def _route_delta(before, after) -> int:
    return after["hits"] + after["misses"] - before["hits"] - before["misses"]


def _shared_rstd(x: torch.Tensor) -> torch.Tensor:
    """Create an FP32 rstd fixture without invoking any RMSNorm FWD path."""

    return torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + EPS).contiguous()


def _tolerance(dtype: torch.dtype, result: str) -> tuple[float, float]:
    if result == "dweight":
        return {
            torch.float32: (1e-4, 1e-2),
            torch.float16: (3e-2, 2e-1),
            torch.bfloat16: (1e-1, 1.0),
        }[dtype]
    return {
        torch.float32: (1e-4, 1e-3),
        torch.float16: (3e-2, 3e-2),
        torch.bfloat16: (1e-1, 2e-1),
    }[dtype]


def _direct_case_name(case) -> str:
    m, n, dtype = case
    return f"M{m}_N{n}_{str(dtype).removeprefix('torch.')}"


def _perf_case_name(case) -> str:
    dtype, n_first, n_last, enabled_m, _ = case
    return f"{str(dtype).removeprefix('torch.')}_N{n_first}_{n_last}_M{enabled_m}"


class TestFlyDSLRMSNormBwdPerfGate(TestCase):
    @parametrize("case", BWD_PERF_RANGES, name_fn=_perf_case_name)
    def test_mi355_wall_to_sync_ranges(self, case):
        from torch._native.ops.norm.flydsl_rmsnorm_bwd_impl import (
            _fused_rms_norm_bwd_perf_wins,
        )

        dtype, n_first, n_last, enabled_m, aten_m = case
        for n in (n_first, n_last):
            enabled = torch.empty((enabled_m, n), device="meta", dtype=dtype)
            self.assertTrue(_fused_rms_norm_bwd_perf_wins(enabled, n))
            fallback = torch.empty((aten_m, n), device="meta", dtype=dtype)
            self.assertFalse(_fused_rms_norm_bwd_perf_wins(fallback, n))

    def test_unprofitable_n_gaps_stay_on_aten(self):
        from torch._native.ops.norm.flydsl_rmsnorm_bwd_impl import (
            _fused_rms_norm_bwd_perf_wins,
        )

        cases = (
            (torch.float16, 64),
            (torch.float16, 512),
            (torch.bfloat16, 64),
            (torch.bfloat16, 512),
            (torch.float32, 64),
            (torch.float32, 128),
        )
        for dtype, n in cases:
            with self.subTest(dtype=dtype, n=n):
                x = torch.empty((65536, n), device="meta", dtype=dtype)
                self.assertFalse(_fused_rms_norm_bwd_perf_wins(x, n))

    def test_n_outside_measured_range_stays_on_aten(self):
        from torch._native.ops.norm.flydsl_rmsnorm_bwd_impl import (
            _fused_rms_norm_bwd_perf_wins,
        )

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for n in (15, 65537):
                with self.subTest(dtype=dtype, n=n):
                    x = torch.empty((32768, n), device="meta", dtype=dtype)
                    self.assertFalse(_fused_rms_norm_bwd_perf_wins(x, n))

        unsupported = torch.empty((32768, 32768), device="meta", dtype=torch.float64)
        self.assertFalse(_fused_rms_norm_bwd_perf_wins(unsupported, 32768))

    def test_bwd_buffer_address_limit(self):
        from torch._native.ops.norm.flydsl_rmsnorm_bwd_impl import (
            _fused_rms_norm_bwd_buffer_addressable,
        )

        for dtype, m, n in (
            (torch.float16, 32768, 65536),
            (torch.bfloat16, 32768, 65536),
            (torch.float32, 16384, 65536),
        ):
            with self.subTest(dtype=dtype):
                below_limit = torch.empty((m // 2, n), device="meta", dtype=dtype)
                at_limit = torch.empty((m, n), device="meta", dtype=dtype)
                self.assertTrue(_fused_rms_norm_bwd_buffer_addressable(below_limit))
                self.assertFalse(_fused_rms_norm_bwd_buffer_addressable(at_limit))


@unittest.skipUnless(TEST_CUDA and torch.version.hip is not None, "ROCm required")
@unittest.skipUnless(
    _flydsl_bwd_registered(), "FlyDSL RMSNorm backward is not registered"
)
class TestFlyDSLRMSNormBwd(TestCase):
    def setUp(self):
        super().setUp()
        from torch._native.ops.norm.flydsl_rmsnorm_bwd import clear_rmsnorm_bwd_caches

        clear_rmsnorm_bwd_caches()
        torch.manual_seed(0)

    def _make_inputs(self, m, n, dtype):
        x = torch.randn((m, n), device="cuda", dtype=dtype)
        weight = torch.randn((n,), device="cuda", dtype=dtype)
        grad_out = torch.randn((m, n), device="cuda", dtype=dtype)
        return x, weight, grad_out

    def _assert_close(self, actual, expected, dtype, result):
        rtol, atol = _tolerance(dtype, result)
        self.assertEqual(actual, expected, rtol=rtol, atol=atol)

    def _call_bwd(self, grad_out, x, rstd, weight, output_mask):
        return torch.ops.aten._fused_rms_norm_backward.default(
            grad_out,
            x,
            [x.shape[-1]],
            rstd,
            weight,
            output_mask,
        )

    @parametrize("case", DIRECT_CASES, name_fn=_direct_case_name)
    def test_direct_backward_matches_aten(self, case):
        from torch._native.ops.norm.flydsl_rmsnorm_bwd import (
            rmsnorm_bwd,
            rmsnorm_bwd_cache_info,
        )

        m, n, dtype = case
        x, weight, grad_out = self._make_inputs(m, n, dtype)
        with torch.inference_mode():
            rstd = _shared_rstd(x)
        with torch.inference_mode(), pn.flydsl.disabled():
            ref_dx, ref_dw = self._call_bwd(grad_out, x, rstd, weight, [True, True])

        with torch.inference_mode():
            got_dx, got_dw = rmsnorm_bwd(grad_out, x, rstd, weight)

        self.assertEqual(rmsnorm_bwd_cache_info().misses, 1)
        self.assertEqual(got_dx.dtype, dtype)
        self.assertEqual(got_dw.dtype, dtype)
        self._assert_close(got_dx, ref_dx, dtype, "dx")
        self._assert_close(got_dw, ref_dw, dtype, "dweight")

    def test_backward_dispatch_reuses_cache_and_masks_fall_back(self):
        m, n, dtype = BWD_DISPATCH_M, BWD_DISPATCH_N, BWD_DISPATCH_DTYPE
        x, weight, grad_out = self._make_inputs(m, n, dtype)
        with torch.inference_mode():
            rstd = _shared_rstd(x)
        with torch.inference_mode(), pn.flydsl.disabled():
            ref_dx, ref_dw = self._call_bwd(grad_out, x, rstd, weight, [True, True])

        for _ in range(2):
            before = _cache_counts()
            with torch.inference_mode():
                got_dx, got_dw = self._call_bwd(grad_out, x, rstd, weight, [True, True])
            self.assertEqual(_route_delta(before, _cache_counts()), 1)
            self._assert_close(got_dx, ref_dx, dtype, "dx")
            self._assert_close(got_dw, ref_dw, dtype, "dweight")

        for output_mask in ([True, False], [False, True]):
            with torch.inference_mode(), pn.flydsl.disabled():
                ref = self._call_bwd(grad_out, x, rstd, weight, output_mask)
            before = _cache_counts()
            with torch.inference_mode():
                got = self._call_bwd(grad_out, x, rstd, weight, output_mask)
            self.assertEqual(_route_delta(before, _cache_counts()), 0)
            for actual, expected, result in zip(got, ref, ("dx", "dweight")):
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    self._assert_close(actual, expected, dtype, result)

    def test_backward_below_perf_threshold_falls_back(self):
        m, n, dtype = 2048, 6144, torch.float16
        x, weight, grad_out = self._make_inputs(m, n, dtype)
        with torch.inference_mode():
            rstd = _shared_rstd(x)
        with torch.inference_mode(), pn.flydsl.disabled():
            ref_dx, ref_dw = self._call_bwd(grad_out, x, rstd, weight, [True, True])

        before = _cache_counts()
        with torch.inference_mode():
            got_dx, got_dw = self._call_bwd(grad_out, x, rstd, weight, [True, True])
        self.assertEqual(_route_delta(before, _cache_counts()), 0)
        self._assert_close(got_dx, ref_dx, dtype, "dx")
        self._assert_close(got_dw, ref_dw, dtype, "dweight")


instantiate_parametrized_tests(TestFlyDSLRMSNormBwdPerfGate)
instantiate_parametrized_tests(TestFlyDSLRMSNormBwd)


if __name__ == "__main__":
    run_tests()
