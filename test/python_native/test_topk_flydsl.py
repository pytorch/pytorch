# Owner(s): ["module: dsl-native-ops"]

import unittest

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


_REGISTER_KS = (2, 4, 8, 16)
_CORRECTNESS_KS = (
    2,
    16,
    64,
    65,
    320,
    350,
    385,
    511,
    704,
    705,
    800,
    832,
    1023,
    1024,
)


def _unsupported_environment_reason() -> str | None:
    if not TEST_CUDA or torch.version.hip is None:
        return "ROCm required"

    from torch._native import flydsl_utils as fu

    if fu.check_native_jit_disabled():
        return "native DSL overrides are disabled via TORCH_DISABLE_NATIVE_JIT"
    if not fu.runtime_available():
        return "FlyDSL runtime is not installed"
    if not fu._version_is_ok():
        return f"FlyDSL {fu.runtime_version()} is outside the supported release"

    from torch._native.ops.topk.flydsl_impl import _is_supported_arch

    if not _is_supported_arch(torch.cuda.current_device()):
        return "FlyDSL TopK override requires gfx950"
    return None


_UNSUPPORTED_REASON = _unsupported_environment_reason()


def _test_n(k: int) -> int:
    from torch._native.ops.topk.flydsl_impl import _radix_n_range, _REGISTER_N_RANGE

    if k in _REGISTER_KS:
        return _REGISTER_N_RANGE[0]
    n_range = _radix_n_range(k)
    if n_range is None:
        raise AssertionError(f"missing radix gate for K={k}")
    return n_range[0]


def _test_m(device_index: int | None = None) -> int:
    from torch._native.ops.topk.flydsl_impl import _min_rows_for_full_wave

    if device_index is None:
        device_index = torch.cuda.current_device()
    return max(256, _min_rows_for_full_wave(device_index))


@unittest.skipIf(_UNSUPPORTED_REASON is not None, str(_UNSUPPORTED_REASON))
class TestFlyDSLTopK(TestCase):
    def setUp(self):
        super().setUp()
        from torch._native.ops.topk.flydsl_kernels import clear_topk_cache

        clear_topk_cache()

    def _make_input(self, *, shape):
        torch.manual_seed(0)
        return make_tensor(shape, device="cuda", dtype=torch.float32)

    def _assert_no_flydsl_compiles(self) -> None:
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        self.assertEqual(topk_cache_info().misses, 0)

    def _assert_topk_matches_aten(self, x: torch.Tensor, k: int) -> None:
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        got_v, got_i = torch.topk(x, k, dim=-1)
        self.assertEqual(topk_cache_info().misses, 1)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)
        if k >= 2:
            diffs = got_v[..., :-1] - got_v[..., 1:]
            self.assertTrue((diffs >= 0).all(), "output is not descending")

    def test_override_is_registered(self):
        operations = pn.get_dsl_operations("flydsl")
        self.assertIn("topk", operations)
        self.assertIn("topk.values", operations)

    @parametrize("k", _CORRECTNESS_KS)
    def test_correctness_random_gaussian(self, k: int):
        torch.manual_seed(0)
        x = make_tensor((_test_m(), _test_n(k)), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", (8, 704))
    def test_correctness_with_extreme_values(self, k: int):
        torch.manual_seed(2)
        x = make_tensor((_test_m(), _test_n(k)), device="cuda", dtype=torch.float32)
        x[:, 0] = float("inf")
        x[:, 1] = float("-inf")
        x[:, 2] = 1e38
        x[:, 3] = -1e38
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", (8, 512))
    def test_correctness_with_nan(self, k: int):
        import struct

        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        torch.manual_seed(10)
        m = _test_m()
        x = make_tensor((m, _test_n(k)), device="cuda", dtype=torch.float32)
        neg_nan = struct.unpack("<f", struct.pack("<I", 0xFFC00000))[0]
        x[:, 0] = float("nan")
        x[:, 1] = neg_nan
        x[:, 2] = float("inf")
        x[:, 3] = float("-inf")
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        got_v, got_i = torch.topk(x, k, dim=-1)
        self.assertEqual(topk_cache_info().misses, 1)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)
        self.assertEqual(got_v.isnan().sum(dim=-1), ref_v.isnan().sum(dim=-1))
        ref_finite = ref_v.masked_select(~ref_v.isnan()).reshape(m, -1)
        got_finite = got_v.masked_select(~got_v.isnan()).reshape(m, -1)
        self.assertEqual(got_finite, ref_finite)

    def test_nd_input(self):
        torch.manual_seed(3)
        k = 704
        n = _test_n(k)
        rows = (_test_m() + 3) // 4
        x = make_tensor((4, rows, n), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)
        got_v, _ = torch.topk(x, k, dim=-1)
        self.assertEqual(got_v.shape, (4, rows, k))

    @parametrize("k", (8, 704))
    def test_out_variant(self, k: int):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        torch.manual_seed(4)
        m = _test_m()
        x = make_tensor((m, _test_n(k)), device="cuda", dtype=torch.float32)
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)

        out_v = torch.empty(m, k, dtype=torch.float32, device="cuda")
        out_i = torch.empty(m, k, dtype=torch.int64, device="cuda")
        got_v, got_i = torch.topk(x, k, dim=-1, out=(out_v, out_i))
        self.assertEqual(topk_cache_info().misses, 1)
        self.assertIs(got_v, out_v)
        self.assertIs(got_i, out_i)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)

    @parametrize("k", (8, 704))
    def test_cow_input_dispatches_and_remains_cow(self, k: int):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        x = self._make_input(shape=(_test_m(), _test_n(k)))
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)

        x = x._lazy_clone()
        self.assertTrue(torch._C._is_cow_tensor(x))
        data_ptr = x.const_data_ptr()

        got_v, got_i = torch.topk(x, k, dim=-1)

        self.assertEqual(topk_cache_info().misses, 1)
        self.assertTrue(torch._C._is_cow_tensor(x))
        self.assertEqual(x.const_data_ptr(), data_ptr)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)

    def test_topk_uses_cache(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        n = _test_n(8)
        m = _test_m()
        x = self._make_input(shape=(m, n))
        with pn.flydsl.disabled():
            ref = torch.topk(x, 8, dim=-1)

        got = torch.topk(x, 8, dim=-1)
        self.assertEqual(got.values, ref.values)
        self.assertEqual(got.indices, ref.indices)

        x3 = self._make_input(shape=(2, (m + 1) // 2, n))
        with pn.flydsl.disabled():
            ref3 = torch.topk(x3, 8, dim=-1)
        got3 = torch.topk(x3, 8, dim=-1)
        self.assertEqual(got3.values, ref3.values)
        self.assertEqual(got3.indices, ref3.indices)

        info = topk_cache_info()
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 1)
        self.assertEqual(info.currsize, 1)

    def test_unsupported_k_falls_through_without_compiling(self):
        torch.manual_seed(5)
        bad_k = 32
        x = make_tensor((_test_m(), 4096), device="cuda", dtype=torch.float32)
        with pn.flydsl.disabled():
            ref = torch.topk(x, bad_k, dim=-1)
        got = torch.topk(x, bad_k, dim=-1)
        self._assert_no_flydsl_compiles()
        self.assertEqual(got.values, ref.values)
        self.assertEqual(got.indices, ref.indices)

    def test_register_non_power_of_two_n_falls_through_without_compiling(self):
        torch.manual_seed(8)
        k = 8
        x = make_tensor((_test_m(), 1537), device="cuda", dtype=torch.float32)
        with pn.flydsl.disabled():
            ref = torch.topk(x, k, dim=-1)
        got = torch.topk(x, k, dim=-1)
        self._assert_no_flydsl_compiles()
        self.assertEqual(got, ref)

    @parametrize("k,n", ((64, 4096), (320, 8192)))
    def test_radix_below_performance_gate_falls_through_without_compiling(
        self, k: int, n: int
    ):
        x = self._make_input(shape=(_test_m(), n))
        with pn.flydsl.disabled():
            ref = torch.topk(x, k, dim=-1)
        got = torch.topk(x, k, dim=-1)
        self._assert_no_flydsl_compiles()
        self.assertEqual(got, ref)

    def test_radix_non_multiple_of_four_n(self):
        torch.manual_seed(11)
        x = make_tensor((_test_m(), 32769), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, 512)

    def test_unsupported_dtype_falls_through_without_compiling(self):
        x = make_tensor((_test_m(), _test_n(8)), device="cuda", dtype=torch.float16)
        with pn.flydsl.disabled():
            ref = torch.topk(x, 8, dim=-1)
        got = torch.topk(x, 8, dim=-1)
        self._assert_no_flydsl_compiles()
        self.assertEqual(got.values, ref.values)
        self.assertEqual(torch.gather(x, -1, got.indices), got.values)

    @parametrize("largest,sorted_", ((False, True), (True, False)))
    def test_unsupported_options_fall_through_without_compiling(
        self, largest: bool, sorted_: bool
    ):
        x = self._make_input(shape=(_test_m(), _test_n(8)))
        with pn.flydsl.disabled():
            ref = torch.topk(x, 8, dim=-1, largest=largest, sorted=sorted_)
        got = torch.topk(x, 8, dim=-1, largest=largest, sorted=sorted_)
        self._assert_no_flydsl_compiles()
        self.assertEqual(torch.gather(x, -1, got.indices), got.values)
        got_values = torch.sort(got.values, descending=largest).values
        ref_values = torch.sort(ref.values, descending=largest).values
        self.assertEqual(got_values, ref_values)

    def test_non_last_dim_falls_through_without_compiling(self):
        x = self._make_input(shape=(_test_m(), _test_n(8)))
        with pn.flydsl.disabled():
            ref = torch.topk(x, 8, dim=0)
        got = torch.topk(x, 8, dim=0)
        self._assert_no_flydsl_compiles()
        self.assertEqual(got, ref)

    def test_noncontiguous_input_falls_through_without_compiling(self):
        base = self._make_input(shape=(_test_n(8), _test_m()))
        x = base.transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        with pn.flydsl.disabled():
            ref = torch.topk(x, 8, dim=-1)
        got = torch.topk(x, 8, dim=-1)
        self._assert_no_flydsl_compiles()
        self.assertEqual(got, ref)

    def test_too_few_rows_falls_through_without_compiling(self):
        from torch._native.ops.topk.flydsl_impl import _min_rows_for_full_wave

        rows = _min_rows_for_full_wave(torch.cuda.current_device()) - 1
        x = self._make_input(shape=(rows, _test_n(8)))
        with pn.flydsl.disabled():
            ref = torch.topk(x, 8, dim=-1)
        got = torch.topk(x, 8, dim=-1)
        self._assert_no_flydsl_compiles()
        self.assertEqual(got, ref)

    def test_noncontiguous_out_dispatches_and_matches_aten(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        k = 8
        m = _test_m()
        x = self._make_input(shape=(m, _test_n(k)))
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)

        values = torch.empty((m, 2 * k), device="cuda", dtype=torch.float32)[:, ::2]
        indices = torch.empty((m, 2 * k), device="cuda", dtype=torch.int64)[:, ::2]
        self.assertFalse(values.is_contiguous())
        self.assertFalse(indices.is_contiguous())

        got_v, got_i = torch.topk(x, k, dim=-1, out=(values, indices))
        self.assertEqual(topk_cache_info().misses, 1)
        self.assertIs(got_v, values)
        self.assertIs(got_i, indices)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)

    @parametrize("k", (64, 257, 704, 1023))
    def test_deterministic_mode_matches_aten_with_heavy_ties(self, k: int):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        torch.manual_seed(6)
        x = torch.randint(
            0, 4, (_test_m(), _test_n(k)), device="cuda", dtype=torch.float32
        )
        with pn.flydsl.disabled():
            ref_v, ref_i = torch.topk(x, k, dim=-1)

        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            v1, i1 = torch.topk(x, k, dim=-1)
            v2, i2 = torch.topk(x, k, dim=-1)
        finally:
            torch.use_deterministic_algorithms(prior)
        self.assertEqual(topk_cache_info().misses, 1)
        self.assertEqual(v1, v2)
        self.assertEqual(i1, i2)
        self.assertEqual(v1, ref_v)
        self.assertEqual(i1, ref_i)

    def test_register_stable_with_heavy_ties(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        torch.manual_seed(9)
        k = 8
        x = torch.randint(
            0, 4, (_test_m(), _test_n(k)), device="cuda", dtype=torch.float32
        )
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        v1, i1 = torch.topk(x, k, dim=-1)
        v2, i2 = torch.topk(x, k, dim=-1)
        self.assertEqual(topk_cache_info().misses, 1)
        self.assertEqual(v1, v2)
        self.assertEqual(i1, i2)
        self.assertEqual(v1, ref_v)
        self.assertEqual(torch.gather(x, -1, i1), v1)

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "requires at least 2 visible CUDA devices",
    )
    def test_non_current_device(self):
        old_device = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            k = 8
            x = make_tensor(
                (_test_m(1), _test_n(k)), device="cuda:1", dtype=torch.float32
            )
            with pn.flydsl.disabled():
                ref_v, _ = torch.topk(x, k, dim=-1)
            got_v, got_i = torch.topk(x, k, dim=-1)

            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(got_v.device, torch.device("cuda:1"))
            self.assertEqual(got_v, ref_v)
            self.assertEqual(torch.gather(x, -1, got_i), got_v)
        finally:
            torch.cuda.set_device(old_device)


instantiate_parametrized_tests(TestFlyDSLTopK)


if __name__ == "__main__":
    run_tests()
