# Owner(s): ["module: dsl-native-ops"]

import unittest

import torch
import torch.backends.python_native as pn
from torch._native.ops.topk.flydsl_impl import _REGISTER_KS as _impl_register_ks
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


# Taken from the gate rather than restated, so widening _REGISTER_KS cannot
# leave these parametrizations silently testing the old set.
_REGISTER_KS = tuple(sorted(_impl_register_ks))
_RADIX_CORRECTNESS_KS = (64, 65, 128, 129, 256, 257, 383, 384, 512, 513, 831, 832, 1024)
_CORRECTNESS_KS = _REGISTER_KS + _RADIX_CORRECTNESS_KS
# One K per radix gate row, plus the register gate, to cover each N ceiling.
_GATE_MAX_N_KS = (max(_REGISTER_KS), 64, 257, 384, 1024)


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


def _gate_n_bounds(k: int) -> tuple[int, int]:
    from torch._native.ops.topk.flydsl_impl import _radix_n_range, _REGISTER_N_BOUNDS

    if k in _REGISTER_KS:
        return _REGISTER_N_BOUNDS
    n_range = _radix_n_range(k)
    if n_range is None:
        raise AssertionError(f"missing radix gate for K={k}")
    return n_range


def _test_n(k: int) -> int:
    return _gate_n_bounds(k)[0]


def _test_n_max(k: int) -> int:
    return _gate_n_bounds(k)[1]


def _special_cols(n: int) -> tuple[int, ...]:
    """Columns to plant special values in.

    Consecutive columns all land in one thread's first vector load, so they
    exercise neither a later tile nor the scalar tail. These are spread across
    the row and include the last column, which is where a tail lives when one
    does.
    """
    return (0, n // 3 + 1, n // 2, 2 * n // 3 + 2, n - 1)


def _expected_kernel(k: int, n: int) -> str:
    """Which specialization the gate must pick for this shape."""
    from torch._native.ops.topk.flydsl_impl import _kernel_for

    kernel = _kernel_for(k, n)
    if kernel is None:
        raise AssertionError(f"K={k} N={n} is outside the FlyDSL gate")
    return kernel


def _test_m(device_index: int | None = None) -> int:
    from torch._native.ops.topk.flydsl_impl import _min_rows_for_full_wave

    if device_index is None:
        device_index = torch.cuda.current_device()
    return max(256, _min_rows_for_full_wave(device_index))


class TestFlyDSLTopKGates(TestCase):
    @parametrize(
        "rows_m,n,itemsize,expected",
        (
            ((1 << 20) - 1, 1024, 4, True),
            (1 << 20, 1024, 4, False),
            (1 << 31, 1, 1, False),
            (1, 1 << 31, 1, False),
        ),
    )
    def test_int32_buffer_span(
        self, rows_m: int, n: int, itemsize: int, expected: bool
    ):
        from torch._native.ops.topk.flydsl_impl import _fits_int32_buffer_span

        self.assertEqual(_fits_int32_buffer_span(rows_m, n, itemsize), expected)


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

        kernel = _expected_kernel(k, x.shape[-1])
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        got_v, got_i = torch.topk(x, k, dim=-1)
        # Pin the specialization, then confirm the other one stayed cold: the
        # summed counter alone would pass even if the gate picked the wrong
        # kernel for this shape.
        self.assertEqual(topk_cache_info(kernel).misses, 1)
        self.assertEqual(topk_cache_info().misses, 1)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)
        if k >= 2:
            self.assertTrue(
                (got_v[..., :-1] >= got_v[..., 1:]).all(),
                "output is not descending",
            )

    @parametrize("k", _CORRECTNESS_KS)
    def test_correctness(self, k: int):
        torch.manual_seed(0)
        x = make_tensor((_test_m(), _test_n(k)), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", (8, 64, 257, 704, 1023))
    def test_correctness_with_duplicates(self, k: int):
        torch.manual_seed(1)
        x = torch.randint(
            0, 50, (_test_m(), _test_n(k)), device="cuda", dtype=torch.float32
        )
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", (8, 704))
    def test_correctness_with_extreme_values(self, k: int):
        torch.manual_seed(2)
        n = _test_n(k)
        x = make_tensor((_test_m(), n), device="cuda", dtype=torch.float32)
        for col, val in zip(
            _special_cols(n), (float("inf"), float("-inf"), 1e38, -1e38, float("inf"))
        ):
            x[:, col] = val
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", (8, 512))
    def test_correctness_with_nan(self, k: int):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        torch.manual_seed(10)
        m = _test_m()
        n = _test_n(k)
        x = make_tensor((m, n), device="cuda", dtype=torch.float32)
        x_bits = x.view(torch.int32)
        cols = _special_cols(n)
        x_bits[:, cols[0]] = 0x7FC12345
        x_bits[:, cols[1]] = 0xFFC54321 - (1 << 32)
        x[:, cols[2]] = float("inf")
        x[:, cols[3]] = float("-inf")
        x_bits[:, cols[4]] = 0x7FC12345
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        got_v, got_i = torch.topk(x, k, dim=-1)
        self.assertEqual(topk_cache_info(_expected_kernel(k, n)).misses, 1)
        gathered = torch.gather(x, -1, got_i)
        self.assertEqual(gathered.view(torch.int32), got_v.view(torch.int32))
        self.assertEqual(got_v.isnan().sum(dim=-1), ref_v.isnan().sum(dim=-1))
        ref_finite = ref_v.masked_select(~ref_v.isnan()).reshape(m, -1)
        got_finite = got_v.masked_select(~got_v.isnan()).reshape(m, -1)
        self.assertEqual(got_finite, ref_finite)

    def test_3d_input(self):
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

    def test_out_values_overlapping_input(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        k = 8
        m = 16 * _test_m()
        n = _test_n(k)
        x = self._make_input(shape=(m, n))
        original = x.clone()
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(original, k, dim=-1)

        values = x.flatten()[m * (n - k) :].view(m, k)
        indices = torch.empty((m, k), device="cuda", dtype=torch.int64)
        self.assertTrue(torch._C._overlaps(x, values))

        got_v, got_i = torch.topk(x, k, dim=-1, out=(values, indices))
        self.assertEqual(topk_cache_info().misses, 1)
        self.assertIs(got_v, values)
        self.assertIs(got_i, indices)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(original, -1, got_i), got_v)

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

    @parametrize("k,n", ((32, 4096), (8, 1537), (64, 4096), (320, 8192)))
    def test_unsupported_configuration_falls_through_without_compiling(
        self, k: int, n: int
    ):
        x = self._make_input(shape=(_test_m(), n))
        torch.topk(x, k, dim=-1)
        self._assert_no_flydsl_compiles()

    def test_radix_non_multiple_of_four_n(self):
        torch.manual_seed(11)
        x = make_tensor((_test_m(), 32769), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, 512)

    @parametrize(
        "case",
        (
            "dtype",
            "smallest",
            "unsorted",
            "non_last_dim",
            "noncontiguous",
            "too_few_rows",
        ),
    )
    def test_unsupported_case_falls_through_without_compiling(self, case: str):
        rows = _test_m()
        if case == "too_few_rows":
            from torch._native.ops.topk.flydsl_impl import _min_rows_for_full_wave

            rows = _min_rows_for_full_wave(torch.cuda.current_device()) - 1
        n = _test_n(8)
        shape = (n, rows) if case == "noncontiguous" else (rows, n)
        dtype = torch.float16 if case == "dtype" else torch.float32
        x = make_tensor(shape, device="cuda", dtype=dtype)
        if case == "noncontiguous":
            x = x.transpose(0, 1)
        dim = 0 if case == "non_last_dim" else -1
        largest = case != "smallest"
        sorted_ = case != "unsorted"
        torch.topk(x, 8, dim=dim, largest=largest, sorted=sorted_)
        self._assert_no_flydsl_compiles()

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

    @parametrize("k", _REGISTER_KS)
    def test_register_tie_order_is_value_desc_index_asc(self, k: int):
        """The register path is used in deterministic mode too, so pin its ties.

        Its keys are ``(ord << 32) | ~idx``, which orders ties
        ``(value desc, idx asc)``. That is reproducible on its own -- which is
        why ``_run`` picks it regardless of the deterministic flag -- but it is
        not aten's small-K index order, so compare against a stable descending
        sort rather than against aten.
        """
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        torch.manual_seed(7)
        n = _test_n(k)
        self.assertEqual(_expected_kernel(k, n), "register")
        # Three distinct values over a wide row: every top-K entry is tied.
        x = torch.randint(0, 3, (_test_m(), n), device="cuda", dtype=torch.float32)
        expected_i = torch.argsort(x, dim=-1, descending=True, stable=True)[:, :k]

        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            v1, i1 = torch.topk(x, k, dim=-1)
            v2, i2 = torch.topk(x, k, dim=-1)
        finally:
            torch.use_deterministic_algorithms(prior)

        self.assertEqual(topk_cache_info("register").misses, 1)
        self.assertEqual(i1, i2)
        self.assertEqual(v1, v2)
        self.assertEqual(i1, expected_i)
        self.assertEqual(torch.gather(x, -1, i1), v1)

    @parametrize("k", _GATE_MAX_N_KS)
    def test_correctness_at_gate_n_ceiling(self, k: int):
        """Cover the top of each gate row, not just the bottom.

        vec_iters, LDS use and the buffer span all scale with N, so the
        ceilings are a different specialization from the floors every other
        test uses.
        """
        torch.manual_seed(12)
        x = make_tensor((_test_m(), _test_n_max(k)), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", (8, 704))
    def test_autograd_passes_through(self, k: int):
        x = make_tensor(
            (_test_m(), _test_n(k)),
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        values, indices = torch.topk(x, k, dim=-1)
        values.sum().backward()
        self.assertIsNotNone(x.grad)
        expected = torch.zeros_like(x)
        expected.scatter_(-1, indices, 1.0)
        self.assertEqual(x.grad, expected)

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "requires at least 2 visible CUDA devices",
    )
    @parametrize("k", (8, 704))
    def test_each_device_gets_its_own_specialization(self, k: int):
        from torch._native.ops.topk.flydsl_impl import _is_supported_arch
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        if not all(_is_supported_arch(index) for index in (0, 1)):
            self.skipTest("requires two gfx950 devices")

        old_device = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            for index in (0, 1):
                device = torch.device("cuda", index)
                x = make_tensor(
                    (_test_m(index), _test_n(k)),
                    device=device,
                    dtype=torch.float32,
                )
                with pn.flydsl.disabled():
                    ref_v, _ = torch.topk(x, k, dim=-1)
                got_v, got_i = torch.topk(x, k, dim=-1)

                self.assertEqual(torch.cuda.current_device(), 0)
                self.assertEqual(got_v.device, device)
                self.assertEqual(got_v, ref_v)
                self.assertEqual(torch.gather(x, -1, got_i), got_v)

            info = topk_cache_info()
            self.assertEqual(info.misses, 2)
            self.assertEqual(info.currsize, 2)
        finally:
            torch.cuda.set_device(old_device)


instantiate_parametrized_tests(TestFlyDSLTopKGates)
instantiate_parametrized_tests(TestFlyDSLTopK)


if __name__ == "__main__":
    run_tests()
