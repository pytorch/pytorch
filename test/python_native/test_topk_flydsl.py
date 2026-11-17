# Owner(s): ["module: dsl-native-ops"]

import unittest

import torch
import torch.backends.python_native as pn
from torch._native import flydsl_utils as fu
from torch._native.ops.topk.flydsl_impl import (
    _REGISTER_KS as _impl_register_ks,
    _SUPPORTED_ARCHES,
    _SUPPORTED_DTYPES,
)
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoFlyDSL,
    TestCase,
)


_REGISTER_KS = tuple(sorted(_impl_register_ks))
# Radix sort-length padding edges; dtype/K-bucket correctness is covered below.
_RADIX_PADDING_KS = (65, 129, 513)
# One K per radix gate row, plus the register gate, to cover each N ceiling.
_GATE_MAX_N_KS = (max(_REGISTER_KS), 64, 257, 513)


def _gate_n_bounds(k: int, dtype: torch.dtype = torch.float32) -> tuple[int, int]:
    from torch._native.ops.topk.flydsl_impl import _radix_n_range, _REGISTER_N_BOUNDS

    if k in _REGISTER_KS:
        n_range = _REGISTER_N_BOUNDS.get(dtype)
        if n_range is None:
            raise AssertionError(f"missing register gate for dtype={dtype}")
        return n_range
    n_range = _radix_n_range(k, dtype)
    if n_range is None:
        raise AssertionError(f"missing radix gate for K={k} dtype={dtype}")
    return n_range


def _test_n(k: int, dtype: torch.dtype = torch.float32) -> int:
    return _gate_n_bounds(k, dtype)[0]


def _test_n_max(k: int, dtype: torch.dtype = torch.float32) -> int:
    return _gate_n_bounds(k, dtype)[1]


def _special_cols(n: int) -> tuple[int, ...]:
    return (0, n // 3 + 1, n // 2, 2 * n // 3 + 2, n - 1)


def _expected_kernel(k: int, n: int, dtype: torch.dtype = torch.float32) -> str:
    from torch._native.ops.topk.flydsl_impl import _kernel_for

    kernel = _kernel_for(k, n, dtype)
    if kernel is None:
        raise AssertionError(f"K={k} N={n} dtype={dtype} is outside the FlyDSL gate")
    return kernel


def _test_m(
    k: int = 8,
    dtype: torch.dtype = torch.float32,
    device_index: int | None = None,
) -> int:
    return max(256, _gate_min_m(k, dtype, device_index))


def _gate_min_m(
    k: int,
    dtype: torch.dtype = torch.float32,
    device_index: int | None = None,
) -> int:
    from torch._native.ops.topk.flydsl_impl import (
        _min_rows_for_full_wave,
        _RADIX_MIN_CU_WAVES,
        _REGISTER_MIN_CU_WAVES,
    )

    if device_index is None:
        device_index = torch.cuda.current_device()
    min_rows = _min_rows_for_full_wave(device_index)
    min_cu_waves = (
        _REGISTER_MIN_CU_WAVES[dtype] if k in _REGISTER_KS else _RADIX_MIN_CU_WAVES
    )
    return min_rows * min_cu_waves


def _make_test_input(shape, dtype: torch.dtype) -> torch.Tensor:
    kwargs = {}
    if not dtype.is_floating_point:
        info = torch.iinfo(dtype)
        kwargs = {"low": info.min, "high": info.max}
    return make_tensor(shape, device="cuda", dtype=dtype, **kwargs)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoFlyDSL
@unittest.skipIf(
    torch.version.hip is None
    or not fu._is_supported_arch(
        torch.cuda.current_device() if torch.cuda.is_available() else 0,
        _SUPPORTED_ARCHES,
    ),
    "FlyDSL TopK override requires gfx950",
)
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

    def _assert_topk_matches_aten(
        self, x: torch.Tensor, k: int, *, largest: bool = True
    ) -> None:
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1, largest=largest)
        got_v, got_i = torch.topk(x, k, dim=-1, largest=largest)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)
        if k >= 2:
            sorted_i = got_i.sort(dim=-1).values
            self.assertTrue(sorted_i.diff(dim=-1).ne(0).all())
        if k >= 2:
            if largest:
                ordered = got_v[..., :-1] >= got_v[..., 1:]
            else:
                ordered = got_v[..., :-1] <= got_v[..., 1:]
            self.assertTrue(
                ordered.all(),
                f"output is not {'descending' if largest else 'ascending'}",
            )

    @parametrize("k", _RADIX_PADDING_KS)
    def test_radix_sort_length_padding(self, k: int):
        torch.manual_seed(0)
        x = make_tensor((_test_m(), _test_n(k)), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)

    def test_register_correctness_with_odd_rows(self):
        # Register kernel packs two rows per CTA; odd M leaves a tail
        # block where in_bounds is false for one row slot (row_safe falls back
        # to row 0 for loads only; writes stay gated on in_bounds).
        k = 8
        n = _test_n(k)
        self.assertEqual(_expected_kernel(k, n), "register")
        rows = _test_m() | 1
        x = make_tensor((rows, n), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)

    def test_supported_dtypes_and_orderings(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        torch.manual_seed(13)
        for dtype in _SUPPORTED_DTYPES:
            for largest in (True, False):
                for k in (2, 8, 65, 256, 257, 513, 1024):
                    n = _test_n(k, dtype)
                    rows = _test_m(k, dtype)
                    kernel = _expected_kernel(k, n, dtype)
                    with self.subTest(dtype=dtype, largest=largest, k=k, n=n):
                        x = _make_test_input((rows, n), dtype)
                        misses = topk_cache_info(kernel).misses
                        self._assert_topk_matches_aten(x, k, largest=largest)
                        self.assertEqual(topk_cache_info(kernel).misses, misses + 1)

    def test_specials_are_inside_the_selected_set(self):
        # A random row hides every planted value except the winning extreme.
        # Fill with the losing extreme so each planted value, and that
        # extreme, is inside the selected K.
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        for dtype in _SUPPORTED_DTYPES:
            for largest in (True, False):
                for k in (8, 65):
                    n = _test_n(k, dtype)
                    m = _test_m(k, dtype)
                    kernel = _expected_kernel(k, n, dtype)
                    with self.subTest(dtype=dtype, largest=largest, k=k):
                        if dtype.is_floating_point:
                            background = float("-inf") if largest else float("inf")
                            tiny = torch.tensor(
                                torch.finfo(dtype).smallest_normal / 2,
                                dtype=dtype,
                                device="cuda",
                            )
                            planted = (
                                float("inf") if largest else float("-inf"),
                                -0.0,
                                0.0,
                                tiny,
                            )
                        else:
                            info = torch.iinfo(dtype)
                            background = info.min if largest else info.max
                            planted = (
                                info.max if largest else info.min,
                                -1,
                                0,
                                1,
                            )
                        x = torch.full((m, n), background, dtype=dtype, device="cuda")
                        for col, value in enumerate(planted):
                            x[:, col] = value
                        if dtype.is_floating_point:
                            bits = x.view(
                                torch.int32 if dtype == torch.float32 else torch.int16
                            )
                            bits[:, 1] = -(1 << (bits.element_size() * 8 - 1))

                        misses = topk_cache_info(kernel).misses
                        got_v, _ = torch.topk(x, k, largest=largest)
                        self.assertEqual(
                            topk_cache_info(kernel).misses, misses + 1
                        )
                        for value in (background, *planted):
                            skip_zero = (
                                dtype.is_floating_point
                                and not isinstance(value, torch.Tensor)
                                and value == 0
                            )
                            if skip_zero:
                                continue
                            self.assertTrue(
                                got_v.eq(value).any(dim=-1).all(),
                                f"{value} missing from the selected set",
                            )
                        if dtype.is_floating_point:
                            got_bits = got_v.view(bits.dtype)
                            neg_zero = -(1 << (got_bits.element_size() * 8 - 1))
                            self.assertTrue(got_bits.eq(neg_zero).any(dim=-1).all())
                            self.assertTrue(got_bits.eq(0).any(dim=-1).all())
                        self._assert_topk_matches_aten(x, k, largest=largest)

    def test_3d_input(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        torch.manual_seed(3)
        k = 704
        for dtype in _SUPPORTED_DTYPES:
            for largest in (True, False):
                n = _test_n(k, dtype)
                rows = (_test_m(k, dtype) + 3) // 4
                with self.subTest(dtype=dtype, largest=largest, n=n):
                    x = make_tensor((4, rows, n), device="cuda", dtype=dtype)
                    misses = topk_cache_info("radix").misses
                    self._assert_topk_matches_aten(x, k, largest=largest)
                    got_v, _ = torch.topk(x, k, dim=-1, largest=largest)
                    self.assertEqual(topk_cache_info("radix").misses, misses + 1)
                    self.assertEqual(got_v.shape, (4, rows, k))

    def test_correctness_with_extreme_values(self):
        k = 8
        torch.manual_seed(2)
        n = _test_n(k)
        x = make_tensor((_test_m(), n), device="cuda", dtype=torch.float32)
        for col, val in zip(
            _special_cols(n), (float("inf"), float("-inf"), 1e38, -1e38, float("inf"))
        ):
            x[:, col] = val
        self._assert_topk_matches_aten(x, k)

    def test_nan_count_and_finite_subsequence_match_aten(self):
        k = 512
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
        gathered = torch.gather(x, -1, got_i)
        self.assertEqual(gathered.view(torch.int32), got_v.view(torch.int32))
        self.assertEqual(got_v.isnan().sum(dim=-1), ref_v.isnan().sum(dim=-1))
        ref_finite = ref_v.masked_select(~ref_v.isnan()).reshape(m, -1)
        got_finite = got_v.masked_select(~got_v.isnan()).reshape(m, -1)
        self.assertEqual(got_finite, ref_finite)

    def test_narrow_nan_payloads_are_reloaded(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        payloads = {
            torch.float16: (0x7E01, 0xFE02 - (1 << 16)),
            torch.bfloat16: (0x7FC1, 0xFFC2 - (1 << 16)),
        }
        for dtype, (positive_nan, negative_nan) in payloads.items():
            for largest in (True, False):
                for k in (8, 65):
                    n = _test_n(k, dtype)
                    m = _test_m(k, dtype)
                    kernel = _expected_kernel(k, n, dtype)
                    with self.subTest(dtype=dtype, largest=largest, k=k):
                        x = torch.empty((m, n), device="cuda", dtype=dtype)
                        x_bits = x.view(torch.int16)
                        x_bits[:, 0::2] = positive_nan
                        x_bits[:, 1::2] = negative_nan

                        misses = topk_cache_info(kernel).misses
                        got_v, got_i = torch.topk(x, k, largest=largest)
                        gathered = torch.gather(x, -1, got_i)

                        self.assertTrue(got_v.isnan().all())
                        self.assertEqual(
                            gathered.view(torch.int16), got_v.view(torch.int16)
                        )
                        sorted_i = got_i.sort(dim=-1).values
                        self.assertTrue(sorted_i.diff(dim=-1).ne(0).all())
                        self.assertEqual(topk_cache_info(kernel).misses, misses + 1)

    def test_narrow_mixed_nan_semantics(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        payloads = {
            torch.float32: (torch.int32, 0x7FC12345, 0xFFC54321 - (1 << 32)),
            torch.float16: (torch.int16, 0x7E01, 0xFE02 - (1 << 16)),
            torch.bfloat16: (torch.int16, 0x7FC1, 0xFFC2 - (1 << 16)),
        }
        k = 65
        for dtype, (bit_dtype, positive_nan, negative_nan) in payloads.items():
            n = _test_n(k, dtype)
            m = _test_m(k, dtype)
            for largest in (True, False):
                with self.subTest(dtype=dtype, largest=largest):
                    misses = topk_cache_info("radix").misses
                    x = _make_test_input((m, n), dtype)
                    cols = _special_cols(n)[:2]
                    x_bits = x.view(bit_dtype)
                    x_bits[:, cols[0]] = positive_nan
                    x_bits[:, cols[1]] = negative_nan

                    got_v, got_i = torch.topk(x, k, largest=largest)
                    gathered = torch.gather(x, -1, got_i)
                    self.assertEqual(
                        gathered.view(bit_dtype), got_v.view(bit_dtype)
                    )
                    sorted_i = got_i.sort(dim=-1).values
                    self.assertTrue(sorted_i.diff(dim=-1).ne(0).all())

                    finite_x = x.clone()
                    finite_x[x.isnan()] = float("-inf") if largest else float("inf")
                    finite_k = k - len(cols) if largest else k
                    with pn.flydsl.disabled():
                        ref_finite, _ = torch.topk(finite_x, finite_k, largest=largest)
                    if largest:
                        self.assertEqual(
                            got_v.masked_select(~got_v.isnan()).reshape(m, finite_k),
                            ref_finite,
                        )
                        self.assertTrue(got_v.isnan().sum(dim=-1).eq(len(cols)).all())
                    else:
                        self.assertFalse(got_v.isnan().any())
                        self.assertEqual(got_v, ref_finite)
                    self.assertEqual(topk_cache_info("radix").misses, misses + 1)

    def test_out_variant(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        k = 704
        m = _test_m()
        for dtype, largest in ((torch.float32, True), (torch.float16, False)):
            with self.subTest(dtype=dtype, largest=largest):
                x = make_tensor((m, _test_n(k)), device="cuda", dtype=dtype)
                with pn.flydsl.disabled():
                    ref_v, _ = torch.topk(x, k, dim=-1, largest=largest)

                out_v = torch.empty(m, k, dtype=dtype, device="cuda")
                out_i = torch.empty(m, k, dtype=torch.int64, device="cuda")
                misses = topk_cache_info("radix").misses
                got_v, got_i = torch.topk(
                    x,
                    k,
                    dim=-1,
                    largest=largest,
                    out=(out_v, out_i),
                )
                self.assertIs(got_v, out_v)
                self.assertIs(got_i, out_i)
                self.assertEqual(got_v, ref_v)
                self.assertEqual(torch.gather(x, -1, got_i), got_v)
                self.assertEqual(topk_cache_info("radix").misses, misses + 1)

    def test_cow_out_dispatches_and_materializes(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        k = 8
        m = _test_m()
        n = _test_n(k)
        x = self._make_input(shape=(m, n))
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)

        values_base = torch.full((m, k), -1.0, device="cuda")
        indices_base = torch.full((m, k), -1, dtype=torch.int64, device="cuda")
        values = values_base._lazy_clone()
        indices = indices_base._lazy_clone()
        if not torch._C._is_cow_tensor(values):
            self.skipTest("allocator does not support COW sharing")
        self.assertTrue(torch._C._is_cow_tensor(values))
        self.assertTrue(torch._C._is_cow_tensor(indices))

        got_v, got_i = torch.topk(x, k, dim=-1, out=(values, indices))

        self.assertEqual(topk_cache_info(_expected_kernel(k, n)).misses, 1)
        self.assertFalse(torch._C._is_cow_tensor(values))
        self.assertFalse(torch._C._is_cow_tensor(indices))
        self.assertEqual(values_base, torch.full_like(values_base, -1.0))
        self.assertEqual(indices_base, torch.full_like(indices_base, -1))
        self.assertIs(got_v, values)
        self.assertIs(got_i, indices)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)

    def test_out_values_overlapping_input(self):
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
        self.assertIs(got_v, values)
        self.assertIs(got_i, indices)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(original, -1, got_i), got_v)

    def test_cow_input_dispatches_and_remains_cow(self):
        k = 8
        x = self._make_input(shape=(_test_m(), _test_n(k)))
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)

        x = x._lazy_clone()
        if not torch._C._is_cow_tensor(x):
            self.skipTest("allocator does not support COW sharing")
        self.assertTrue(torch._C._is_cow_tensor(x))
        data_ptr = x.const_data_ptr()

        got_v, got_i = torch.topk(x, k, dim=-1)

        self.assertTrue(torch._C._is_cow_tensor(x))
        self.assertEqual(x.const_data_ptr(), data_ptr)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)

    def test_topk_uses_cache(self):
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

        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        info = topk_cache_info("register")
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

    @parametrize(
        "case",
        (
            "unsupported_dtype",
            "unsorted",
            "non_last_dim",
            "noncontiguous",
            "too_few_rows",
        ),
    )
    def test_unsupported_case_falls_through_without_compiling(self, case: str):
        rows = _test_m()
        if case == "too_few_rows":
            rows = _gate_min_m(8) - 1
        n = _test_n(8)
        shape = (n, rows) if case == "noncontiguous" else (rows, n)
        dtype = torch.float64 if case == "unsupported_dtype" else torch.float32
        x = make_tensor(shape, device="cuda", dtype=dtype)
        if case == "noncontiguous":
            x = x.transpose(0, 1)
        dim = 0 if case == "non_last_dim" else -1
        sorted_ = case != "unsorted"
        torch.topk(x, 8, dim=dim, sorted=sorted_)
        self._assert_no_flydsl_compiles()

    def test_radix_non_multiple_of_four_n(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        torch.manual_seed(11)
        k = 512
        for dtype in _SUPPORTED_DTYPES:
            for largest in (True, False):
                n = _test_n(k, dtype) + 1
                m = _test_m(k, dtype)
                with self.subTest(dtype=dtype, largest=largest, n=n):
                    self.assertEqual(_expected_kernel(k, n, dtype), "radix")
                    x = _make_test_input((m, n), dtype)
                    misses = topk_cache_info("radix").misses
                    self._assert_topk_matches_aten(x, k, largest=largest)
                    self.assertEqual(topk_cache_info("radix").misses, misses + 1)

    def test_misaligned_register_base_matches_aten(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        torch.manual_seed(12)
        k = 8
        for dtype in _SUPPORTED_DTYPES:
            for largest in (True, False):
                n = _test_n(k, dtype)
                m = _test_m(k, dtype)
                with self.subTest(dtype=dtype, largest=largest, n=n):
                    self.assertEqual(_expected_kernel(k, n, dtype), "register")
                    buf = _make_test_input((m * n + 1,), dtype)
                    x = buf[1:].reshape(m, n)
                    if not x.is_contiguous():
                        raise AssertionError("expected contiguous misaligned-base view")
                    if x.data_ptr() % 16 == 0:
                        raise AssertionError("expected misaligned base pointer")
                    misses = topk_cache_info("register").misses
                    self._assert_topk_matches_aten(x, k, largest=largest)
                    self.assertEqual(topk_cache_info("register").misses, misses + 1)

    def test_misaligned_radix_base_matches_aten(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        torch.manual_seed(14)
        k = 64
        for dtype in _SUPPORTED_DTYPES:
            for largest in (True, False):
                n = _test_n(k, dtype)
                m = _test_m(k, dtype)
                with self.subTest(dtype=dtype, largest=largest, n=n):
                    self.assertEqual(_expected_kernel(k, n, dtype), "radix")
                    buf = _make_test_input((m * n + 1,), dtype)
                    x = buf[1:].reshape(m, n)
                    if not x.is_contiguous():
                        raise AssertionError("expected contiguous misaligned-base view")
                    if x.data_ptr() % 16 == 0:
                        raise AssertionError("expected misaligned base pointer")
                    misses = topk_cache_info("radix").misses
                    self._assert_topk_matches_aten(x, k, largest=largest)
                    self.assertEqual(topk_cache_info("radix").misses, misses + 1)

    def test_noncontiguous_out_dispatches_and_matches_aten(self):
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
        self.assertIs(got_v, values)
        self.assertIs(got_i, indices)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)

    @parametrize("k", (64, 257, 704, 1023))
    def test_deterministic_mode_matches_aten_with_heavy_ties(self, k: int):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        torch.manual_seed(6)
        n = _test_n(k)
        self.assertEqual(_expected_kernel(k, n), "radix")
        x = torch.randint(0, 4, (_test_m(), n), device="cuda", dtype=torch.float32)

        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with pn.flydsl.disabled():
                ref_v, ref_i = torch.topk(x, k, dim=-1)
            v1, i1 = torch.topk(x, k, dim=-1)
            v2, i2 = torch.topk(x, k, dim=-1)
        finally:
            torch.use_deterministic_algorithms(prior)

        self.assertEqual(topk_cache_info("radix").misses, 1)
        self.assertEqual(v1, v2)
        self.assertEqual(i1, i2)
        self.assertEqual(v1, ref_v)
        self.assertEqual(i1, ref_i)

    @parametrize("k", _REGISTER_KS)
    def test_register_tie_order_is_value_desc_index_asc(self, k: int):
        torch.manual_seed(7)
        n = _test_n(k)
        self.assertEqual(_expected_kernel(k, n), "register")
        x = torch.randint(0, 3, (_test_m(), n), device="cuda", dtype=torch.float32)
        expected_i = torch.argsort(x, dim=-1, descending=True, stable=True)[:, :k]

        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            v1, i1 = torch.topk(x, k, dim=-1)
            v2, i2 = torch.topk(x, k, dim=-1)
        finally:
            torch.use_deterministic_algorithms(prior)

        self.assertEqual(i1, i2)
        self.assertEqual(v1, v2)
        self.assertEqual(i1, expected_i)
        self.assertEqual(torch.gather(x, -1, i1), v1)

    @parametrize("k", _GATE_MAX_N_KS)
    def test_correctness_at_gate_n_ceiling(self, k: int):
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
    def test_each_device_gets_its_own_specialization(self):
        k = 704
        if not all(fu._is_supported_arch(index, _SUPPORTED_ARCHES) for index in (0, 1)):
            self.skipTest("requires two gfx950 devices")

        old_device = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            for index in (0, 1):
                device = torch.device("cuda", index)
                x = make_tensor(
                    (_test_m(k, device_index=index), _test_n(k)),
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

            from torch._native.ops.topk.flydsl_kernels import topk_cache_info

            info = topk_cache_info()
            self.assertEqual(info.misses, 2)
            self.assertEqual(info.currsize, 2)
        finally:
            torch.cuda.set_device(old_device)


instantiate_parametrized_tests(TestFlyDSLTopK)


if __name__ == "__main__":
    run_tests()
