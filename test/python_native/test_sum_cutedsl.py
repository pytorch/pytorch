# Owner(s): ["module: dsl-native-ops"]
#
# Correctness + bitwise-equivalence tests for the CuTeDSL ``aten::sum``
# inner-tree override (``torch/_native/ops/reductions``). The override is gated on
# ``PYTORCH_SUM_INNER_TREE`` and registered for CUDA; these tests are gated on
# CuTeDSL availability so the assertions exercise the CuTeDSL kernel.
#
# The bitwise tests pin the reduction output to precomputed SHA-256 hashes of
# deterministic row sums. The hashes are the same ones the (now removed) ATen
# CUDA inner-tree kernel produced -- the CuTeDSL kernel reproduces them
# bit-for-bit, which is the contract for this op.

import contextlib
import os
import unittest
from unittest import mock

import torch
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FP8,
    SM100OrLater,
    TEST_CUDA,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    skipIfRocm,
    TestCase,
)


def _cutedsl_impl():
    from torch._native.ops.reductions import cutedsl_impl

    return cutedsl_impl


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestSumCuteDSLOverride(TestCase):
    """Tests for the CuTeDSL inner-tree reduction sum override."""

    _ACC_DTYPES = {
        torch.float8_e4m3fn: torch.float32,
        torch.float16: torch.float32,
        torch.float32: torch.float32,
        torch.float64: torch.float64,
    }

    _SHAPES = [
        (1, 32),
        (1, 8192),
        (1, 16384),
        (128, 32),
        (128, 8192),
        (128, 16384),
        (4096, 32),
        (4096, 8192),
        (4096, 16384),
        (1, 27),
        (1, 6561),
        (1, 19683),
        (128, 27),
        (128, 6561),
        (128, 19683),
        (4096, 27),
        (4096, 6561),
        (4096, 19683),
        (1, 36),
        (1, 7776),
        (1, 46656),
        (128, 36),
        (128, 7776),
        (128, 46656),
        (4096, 36),
        (4096, 7776),
        (4096, 46656),
    ]

    # fmt: off
    _EXPECTED = {
        ("float8_e4m3fn", 1, 32): "df3f619804a92fdb",
        ("float8_e4m3fn", 1, 8192): "df3f619804a92fdb",
        ("float8_e4m3fn", 1, 16384): "df3f619804a92fdb",
        ("float8_e4m3fn", 128, 32): "f428ef9e3fbd9469",
        ("float8_e4m3fn", 128, 8192): "4834001f0135cd23",
        ("float8_e4m3fn", 128, 16384): "76e9b171932d1f9f",
        ("float8_e4m3fn", 4096, 32): "a5c21039e21a5323",
        ("float8_e4m3fn", 4096, 8192): "e34643d4c484ef9c",
        ("float8_e4m3fn", 4096, 16384): "289c5f23e410c913",
        ("float8_e4m3fn", 1, 27): "c68830a25204a09f",
        ("float8_e4m3fn", 1, 6561): "c68830a25204a09f",
        ("float8_e4m3fn", 1, 19683): "c68830a25204a09f",
        ("float8_e4m3fn", 128, 27): "2252a0a2c19ada94",
        ("float8_e4m3fn", 128, 6561): "6bf0fe73bdad39f5",
        ("float8_e4m3fn", 128, 19683): "03b2636d8210d4d0",
        ("float8_e4m3fn", 4096, 27): "41896635bfcb0c7c",
        ("float8_e4m3fn", 4096, 6561): "990ad1490c767a17",
        ("float8_e4m3fn", 4096, 19683): "c3cc65b050631480",
        ("float8_e4m3fn", 1, 36): "df3f619804a92fdb",
        ("float8_e4m3fn", 1, 7776): "df3f619804a92fdb",
        ("float8_e4m3fn", 1, 46656): "df3f619804a92fdb",
        ("float8_e4m3fn", 128, 36): "927688b93e357722",
        ("float8_e4m3fn", 128, 7776): "e46696a8a4613062",
        ("float8_e4m3fn", 128, 46656): "a10c4d305d19a761",
        ("float8_e4m3fn", 4096, 36): "efe1cee0d5e0bf4e",
        ("float8_e4m3fn", 4096, 7776): "d5588816762acbd4",
        ("float8_e4m3fn", 4096, 46656): "27fc2749145207e3",
        ("float16", 1, 32): "df3f619804a92fdb",
        ("float16", 1, 8192): "df3f619804a92fdb",
        ("float16", 1, 16384): "df3f619804a92fdb",
        ("float16", 128, 32): "f428ef9e3fbd9469",
        ("float16", 128, 8192): "4834001f0135cd23",
        ("float16", 128, 16384): "76e9b171932d1f9f",
        ("float16", 4096, 32): "a5c21039e21a5323",
        ("float16", 4096, 8192): "e34643d4c484ef9c",
        ("float16", 4096, 16384): "289c5f23e410c913",
        ("float16", 1, 27): "c68830a25204a09f",
        ("float16", 1, 6561): "c68830a25204a09f",
        ("float16", 1, 19683): "c68830a25204a09f",
        ("float16", 128, 27): "2252a0a2c19ada94",
        ("float16", 128, 6561): "6bf0fe73bdad39f5",
        ("float16", 128, 19683): "03b2636d8210d4d0",
        ("float16", 4096, 27): "41896635bfcb0c7c",
        ("float16", 4096, 6561): "990ad1490c767a17",
        ("float16", 4096, 19683): "c3cc65b050631480",
        ("float16", 1, 36): "df3f619804a92fdb",
        ("float16", 1, 7776): "df3f619804a92fdb",
        ("float16", 1, 46656): "df3f619804a92fdb",
        ("float16", 128, 36): "927688b93e357722",
        ("float16", 128, 7776): "e46696a8a4613062",
        ("float16", 128, 46656): "a10c4d305d19a761",
        ("float16", 4096, 36): "efe1cee0d5e0bf4e",
        ("float16", 4096, 7776): "d5588816762acbd4",
        ("float16", 4096, 46656): "27fc2749145207e3",
        ("float32", 1, 32): "df3f619804a92fdb",
        ("float32", 1, 8192): "df3f619804a92fdb",
        ("float32", 1, 16384): "df3f619804a92fdb",
        ("float32", 128, 32): "f428ef9e3fbd9469",
        ("float32", 128, 8192): "4834001f0135cd23",
        ("float32", 128, 16384): "76e9b171932d1f9f",
        ("float32", 4096, 32): "a5c21039e21a5323",
        ("float32", 4096, 8192): "e34643d4c484ef9c",
        ("float32", 4096, 16384): "289c5f23e410c913",
        ("float32", 1, 27): "c68830a25204a09f",
        ("float32", 1, 6561): "c68830a25204a09f",
        ("float32", 1, 19683): "c68830a25204a09f",
        ("float32", 128, 27): "2252a0a2c19ada94",
        ("float32", 128, 6561): "6bf0fe73bdad39f5",
        ("float32", 128, 19683): "03b2636d8210d4d0",
        ("float32", 4096, 27): "41896635bfcb0c7c",
        ("float32", 4096, 6561): "990ad1490c767a17",
        ("float32", 4096, 19683): "c3cc65b050631480",
        ("float32", 1, 36): "df3f619804a92fdb",
        ("float32", 1, 7776): "df3f619804a92fdb",
        ("float32", 1, 46656): "df3f619804a92fdb",
        ("float32", 128, 36): "927688b93e357722",
        ("float32", 128, 7776): "e46696a8a4613062",
        ("float32", 128, 46656): "a10c4d305d19a761",
        ("float32", 4096, 36): "efe1cee0d5e0bf4e",
        ("float32", 4096, 7776): "d5588816762acbd4",
        ("float32", 4096, 46656): "27fc2749145207e3",
        ("float64", 1, 32): "af5570f5a1810b7a",
        ("float64", 1, 8192): "af5570f5a1810b7a",
        ("float64", 1, 16384): "af5570f5a1810b7a",
        ("float64", 128, 32): "35a0993dedd64659",
        ("float64", 128, 8192): "05102af99bde7f20",
        ("float64", 128, 16384): "c54af6a4a43a21d4",
        ("float64", 4096, 32): "a5e02a44cc2dd361",
        ("float64", 4096, 8192): "e9ba5b9f0409fec2",
        ("float64", 4096, 16384): "6d8144f96bb8a210",
        ("float64", 1, 27): "e77817b649821c63",
        ("float64", 1, 6561): "e77817b649821c63",
        ("float64", 1, 19683): "e77817b649821c63",
        ("float64", 128, 27): "86e9b1f2d6fd4504",
        ("float64", 128, 6561): "e761055386fc7295",
        ("float64", 128, 19683): "3ce4cdd14e0b152f",
        ("float64", 4096, 27): "4926c5d5541728d1",
        ("float64", 4096, 6561): "417525b3b5d3de87",
        ("float64", 4096, 19683): "cd8ef6c48ad2c67f",
        ("float64", 1, 36): "af5570f5a1810b7a",
        ("float64", 1, 7776): "af5570f5a1810b7a",
        ("float64", 1, 46656): "af5570f5a1810b7a",
        ("float64", 128, 36): "f00a064b84ba50b4",
        ("float64", 128, 7776): "c227b341b6240d30",
        ("float64", 128, 46656): "af1f2b32fdaf1f62",
        ("float64", 4096, 36): "5cb245154bac020c",
        ("float64", 4096, 7776): "751916d8d6b9dcba",
        ("float64", 4096, 46656): "498bc246ba0b069a",
    }
    # fmt: on

    _DTYPE_MAP = {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    @contextlib.contextmanager
    def _inner_tree_flag(self):
        old_value = os.environ.get("PYTORCH_SUM_INNER_TREE")
        os.environ["PYTORCH_SUM_INNER_TREE"] = "1"
        try:
            yield
        finally:
            if old_value is None:
                os.environ.pop("PYTORCH_SUM_INNER_TREE", None)
            else:
                os.environ["PYTORCH_SUM_INNER_TREE"] = old_value

    def _make_input(self, m, n, dtype):
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        cols = torch.arange(n, device="cuda", dtype=compute_dtype)
        values = ((cols % 2) * 2 - 1).reshape(1, n)
        if m > 1:
            rows = torch.arange(m, device="cuda", dtype=compute_dtype).reshape(m, 1)
            values = values + ((rows % 5) - 2) / 4
        return values.to(dtype)

    def _eager_sum(self, x):
        acc_dtype = self._ACC_DTYPES[x.dtype]
        if x.dtype == torch.float8_e4m3fn:
            return x.to(torch.float32).sum(dim=1).to(acc_dtype)
        return x.to(acc_dtype).sum(dim=1)

    def _sha(self, t):
        import hashlib

        return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()[:16]

    def _make_order_sensitive_input(self, m, n, dtype):
        values = torch.arange(m * n, device="cuda", dtype=torch.float64).reshape(m, n)
        return (((values % 29) - 14) / 29 + ((values % 7) - 3) / 13).to(dtype)

    # --- predicate accept/reject ---

    def test_cond_accepts_covered_shapes(self):
        impl = _cutedsl_impl()
        with self._inner_tree_flag():
            for dtype in (torch.float32, torch.float64):
                x = torch.randn(128, 8192, device="cuda", dtype=dtype)
                self.assertTrue(impl._cond(x, [1]))
                # Strided outer input (contiguous reduced dim) is still covered.
                strided = torch.randn(256, 8192, device="cuda", dtype=dtype)[::2]
                self.assertTrue(impl._cond(strided, [1]))

    def test_cond_rejects_unsupported(self):
        impl = _cutedsl_impl()
        x = torch.randn(128, 8192, device="cuda", dtype=torch.float32)
        # No flag -> reject.
        self.assertFalse(impl._cond(x, [1]))
        with self._inner_tree_flag():
            # Multi-dim and full reductions are out of scope.
            self.assertFalse(impl._cond(x, [0, 1]))
            self.assertFalse(impl._cond(x, None))
            # dtype-casting sum is out of scope.
            self.assertFalse(impl._cond(x, [1], dtype=torch.float64))
            # Non-contiguous reduced dim.
            self.assertFalse(impl._cond(x[:, ::2], [1]))
            # Integer / complex dtypes fall through to aten.
            self.assertFalse(
                impl._cond(torch.ones(17, 8192, device="cuda", dtype=torch.int64), [1])
            )
            self.assertFalse(
                impl._cond(
                    torch.ones(17, 8192, device="cuda", dtype=torch.complex64), [1]
                )
            )

    # --- correctness ---

    @parametrize("dtype", [torch.float32, torch.float64])
    def test_override_matches_aten(self, dtype):
        x = self._make_order_sensitive_input(128, 8192, dtype)
        with torch.backends.python_native.cutedsl.disabled():
            ref = x.sum(dim=1)
        with self._inner_tree_flag():
            got = x.sum(dim=1)
        self.assertEqual(got, ref)

    def test_override_out_variant(self):
        x = self._make_order_sensitive_input(128, 8192, torch.float32)
        with torch.backends.python_native.cutedsl.disabled():
            ref = x.sum(dim=1)
        out = torch.empty(128, device="cuda", dtype=torch.float32)
        with self._inner_tree_flag():
            torch.sum(x, dim=1, out=out)
        self.assertEqual(out, ref)

    def test_strided_outer_input(self):
        base = torch.ones(256, 32, device="cuda", dtype=torch.float32)
        base[1::2, :] = 5
        x = base[::2, :]
        with self._inner_tree_flag():
            result = x.sum(dim=1)
        self.assertEqual(result, torch.full((128,), 32, device="cuda", dtype=x.dtype))

    def test_looped_kernel_partial_last_block(self):
        x = torch.ones(129, 256, device="cuda", dtype=torch.float32)
        with self._inner_tree_flag():
            result = x.sum(dim=1)
        self.assertEqual(result, torch.full((129,), 256, device="cuda", dtype=x.dtype))

    @parametrize("dtype", [torch.int64, torch.complex64])
    def test_integer_and_complex_fall_through(self, dtype):
        # CuTeDSL declines integer/complex; the call must fall through to aten.
        x = torch.ones(17, 8192, device="cuda", dtype=dtype)
        with self._inner_tree_flag():
            result = x.sum(dim=1)
        self.assertEqual(result, torch.full((17,), 8192, device="cuda", dtype=dtype))

    # --- bitwise equivalence ---

    @skipIfRocm
    @unittest.skipUnless(
        SM100OrLater, "SM100 hash is generated on GB200 with CUDA 12.8"
    )
    def test_cross_warp_order_sensitive_hash_sm100(self):
        x = self._make_order_sensitive_input(17, 8192, torch.float32)
        with self._inner_tree_flag():
            result = x.sum(dim=1)
        self.assertEqual(self._sha(result), "75d8b1a702344e90")

    @skipIfRocm
    def test_bitwise(self):
        for dtype_name, dtype in self._DTYPE_MAP.items():
            if dtype == torch.float8_e4m3fn and not PLATFORM_SUPPORTS_FP8:
                continue
            for m, n in self._SHAPES:
                with self.subTest(dtype_name=dtype_name, m=m, n=n):
                    x = self._make_input(m, n, dtype)
                    with self._inner_tree_flag():
                        result = self._eager_sum(x)
                    sha = self._sha(result)
                    expected = self._EXPECTED[(dtype_name, m, n)]
                    self.assertEqual(
                        sha,
                        expected,
                        f"{dtype_name} m={m} n={n}: got {sha}, expected {expected}",
                    )

    # --- prod (shares the inner-tree geometry/eligibility with sum) ---

    def _make_prod_input(self, m, n, dtype):
        # Perturb each element by O(1)/n so the length-``n`` product stays
        # bounded (~e^O(1)) for any ``n`` -- otherwise a per-row factor raised
        # to the n-th power overflows and fp reorder error swamps the tolerance.
        # Still order-sensitive (alternating sign + per-row shift).
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        cols = torch.arange(n, device="cuda", dtype=compute_dtype).reshape(1, n)
        pattern = (cols % 2) * 2 - 1  # +/-1 alternating
        if m > 1:
            rows = torch.arange(m, device="cuda", dtype=compute_dtype).reshape(m, 1)
            pattern = pattern + ((rows % 5) - 2)  # per-row shift in [-2, 2]
        return (1.0 + pattern / n).to(dtype)

    @parametrize("dtype", [torch.float32, torch.float64])
    def test_prod_override_matches_aten(self, dtype):
        # A length-n product accumulates ~n*eps of rounding, so inner-tree and
        # ATen (different orders) diverge by that much -- both track the fp64
        # truth equally well, so use a product-appropriate tolerance.
        rtol, atol = (2e-3, 2e-3) if dtype == torch.float32 else (1e-9, 1e-12)
        # Covers the multirow, looped, and two-kernel prod paths.
        for m, n in [(128, 32), (128, 8192), (8, 200000)]:
            with self.subTest(m=m, n=n):
                x = self._make_prod_input(m, n, dtype)
                with torch.backends.python_native.cutedsl.disabled():
                    ref = x.prod(dim=1)
                with self._inner_tree_flag():
                    got = x.prod(dim=1)
                self.assertEqual(got, ref, rtol=rtol, atol=atol)

    def test_prod_override_engaged(self):
        # Proves the CuTeDSL prod override actually executes: its inner-tree
        # order differs bit-for-bit from ATen's default order for an
        # order-sensitive input. If the override silently fell through to ATen,
        # ``got`` would equal ``ref`` and this fails -- guarding the dispatch.
        x = self._make_prod_input(128, 8192, torch.float32)
        with torch.backends.python_native.cutedsl.disabled():
            ref = x.prod(dim=1)
        with self._inner_tree_flag():
            got = x.prod(dim=1)
        self.assertFalse(torch.equal(got, ref))

    @skipIfRocm
    def test_prod_output_hash(self):
        cases = [
            (torch.float32, 128, 8192, "37bd2a1dc47b71c5"),
            (torch.float64, 8, 200000, "374e728d5c174437"),
        ]
        for dtype, m, n, expected in cases:
            with self.subTest(dtype=dtype, m=m, n=n):
                x = self._make_prod_input(m, n, dtype)
                with self._inner_tree_flag():
                    result = x.prod(dim=1)
                self.assertEqual(self._sha(result), expected)

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_prod_low_precision_matches_aten(self, dtype):
        # 0.75 and 1.25 are exactly representable in fp16/bf16; a sparse set of
        # them keeps the length-n product bounded and non-1.0 (a near-1 input
        # collapses to 1.0 at low precision, giving a vacuous test). fp16/bf16
        # are functionally supported but not part of the bitwise contract, so
        # compare to ATen with a low-precision tolerance. Covers the multirow,
        # looped, and two-kernel paths.
        from torch._native.ops.reductions import inner_tree_kernel

        for m, n in [(64, 32), (8, 4096), (8, 65536)]:
            with self.subTest(m=m, n=n):
                x = torch.ones(m, n, device="cuda", dtype=dtype)
                idx = torch.linspace(0, n - 1, 8, dtype=torch.long)
                x[:, idx[0::2]] = 0.75
                x[:, idx[1::2]] = 1.25
                with torch.backends.python_native.cutedsl.disabled():
                    ref = x.prod(dim=1)
                with (
                    mock.patch.object(
                        inner_tree_kernel,
                        "inner_tree_prod_into",
                        wraps=inner_tree_kernel.inner_tree_prod_into,
                    ) as prod_into,
                    self._inner_tree_flag(),
                ):
                    got = x.prod(dim=1)
                # Proves fp16/bf16 route through the CuTeDSL override, not a
                # silent ATen fall-through (which a tolerance compare misses).
                self.assertEqual(prod_into.call_count, 1)
                self.assertEqual(got, ref, rtol=2e-2, atol=2e-2)

    def test_prod_override_out_variant(self):
        x = self._make_prod_input(128, 8192, torch.float32)
        with self._inner_tree_flag():
            ref = x.prod(dim=1)
            out = torch.empty(128, device="cuda", dtype=torch.float32)
            torch.prod(x, dim=1, out=out)
        # The out= path runs the same kernel as the functional path.
        self.assertEqual(out, ref, rtol=0, atol=0)

    def test_prod_strided_outer_input(self):
        # Ragged tails must pad with the multiplicative identity (1), not 0;
        # otherwise every row collapses to 0.
        base = torch.ones(256, 32, device="cuda", dtype=torch.float32)
        base[1::2, :] = 5
        x = base[::2, :]
        with self._inner_tree_flag():
            result = x.prod(dim=1)
        self.assertEqual(result, torch.ones(128, device="cuda", dtype=x.dtype))

    def test_prod_looped_partial_last_block(self):
        x = torch.ones(129, 256, device="cuda", dtype=torch.float32)
        with self._inner_tree_flag():
            result = x.prod(dim=1)
        self.assertEqual(result, torch.ones(129, device="cuda", dtype=x.dtype))

    def test_prod_deterministic(self):
        x = self._make_prod_input(64, 12000, torch.float32)
        with self._inner_tree_flag():
            first = x.prod(dim=1)
            second = x.prod(dim=1)
        self.assertEqual(first, second, rtol=0, atol=0)

    @parametrize("dtype", [torch.int64, torch.complex64])
    def test_prod_integer_and_complex_fall_through(self, dtype):
        x = torch.ones(17, 8192, device="cuda", dtype=dtype)
        with self._inner_tree_flag():
            result = x.prod(dim=1)
        self.assertEqual(result, torch.ones(17, device="cuda", dtype=dtype))

    # --- nansum (shares the inner-tree sum DAG after filtering loaded NaNs) ---

    def _make_nansum_input(self, m, n, dtype):
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        cols = torch.arange(n, device="cuda", dtype=compute_dtype).reshape(1, n)
        rows = torch.arange(m, device="cuda", dtype=compute_dtype).reshape(m, 1)
        values = (cols % 9) - 4 + (rows % 3) - 1
        values[:, ::17] = float("nan")
        return values.to(dtype)

    @parametrize("dtype", [torch.float32, torch.float64])
    def test_nansum_override_matches_aten(self, dtype):
        # Exact small integers keep every fp32 partial sum below 2**24, making
        # the reference independent of its different reduction order.
        for m, n in [(128, 15), (128, 8195), (8, 200003)]:
            with self.subTest(m=m, n=n):
                x = self._make_nansum_input(m, n, dtype)
                with torch.backends.python_native.cutedsl.disabled():
                    ref = torch.nansum(x, dim=1)
                with self._inner_tree_flag():
                    got = torch.nansum(x, dim=1)
                self.assertEqual(got, ref, rtol=0, atol=0)

    @parametrize("dtype", [torch.float32, torch.float64])
    def test_nansum_matches_sum_of_nan_to_num(self, dtype):
        view_dtype = torch.int32 if dtype == torch.float32 else torch.int64
        for m, n in [(128, 15), (128, 8195), (8, 200003)]:
            with self.subTest(m=m, n=n):
                x = self._make_order_sensitive_input(m, n, dtype)
                x[:, ::17] = float("nan")
                zero = torch.zeros((), device=x.device, dtype=x.dtype)
                with self._inner_tree_flag():
                    got = torch.nansum(x, dim=1)
                    expected = torch.sum(torch.where(torch.isnan(x), zero, x), dim=1)
                self.assertTrue(
                    torch.equal(got.view(view_dtype), expected.view(view_dtype))
                )

    def test_nansum_override_engaged(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_nansum_input(128, 8195, torch.float32)
        with torch.backends.python_native.cutedsl.disabled():
            ref = torch.nansum(x, dim=1)
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_nansum_into",
                wraps=inner_tree_kernel.inner_tree_nansum_into,
            ) as nansum_into,
            self._inner_tree_flag(),
        ):
            got = torch.nansum(x, dim=1)
        self.assertEqual(nansum_into.call_count, 1)
        self.assertEqual(got, ref, rtol=0, atol=0)

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_nansum_low_precision_matches_aten(self, dtype):
        from torch._native.ops.reductions import inner_tree_kernel

        for m, n in [(64, 32), (8, 4096), (8, 65536)]:
            with self.subTest(m=m, n=n):
                x = self._make_nansum_input(m, n, dtype)
                with torch.backends.python_native.cutedsl.disabled():
                    ref = torch.nansum(x, dim=1)
                with (
                    mock.patch.object(
                        inner_tree_kernel,
                        "inner_tree_nansum_into",
                        wraps=inner_tree_kernel.inner_tree_nansum_into,
                    ) as nansum_into,
                    self._inner_tree_flag(),
                ):
                    got = torch.nansum(x, dim=1)
                self.assertEqual(nansum_into.call_count, 1)
                self.assertEqual(got, ref, rtol=2e-2, atol=2e-2)

    def test_nansum_override_out_variant(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(128, 8195, torch.float32)
        x[:, ::17] = float("nan")
        with self._inner_tree_flag():
            functional = torch.nansum(x, dim=1)
        out = torch.empty(128, device="cuda", dtype=torch.float32)
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_nansum_into",
                wraps=inner_tree_kernel.inner_tree_nansum_into,
            ) as nansum_into,
            self._inner_tree_flag(),
        ):
            returned = torch.nansum(x, dim=1, out=out)
        self.assertEqual(nansum_into.call_count, 1)
        self.assertIs(returned, out)
        self.assertTrue(
            torch.equal(out.view(torch.int32), functional.view(torch.int32))
        )

    def test_nansum_nan_to_zero(self):
        n = 32769
        x = torch.zeros(3, n, device="cuda", dtype=torch.float32)
        x[0].fill_(float("nan"))
        x[1, 0] = 1.0
        x[1, 3:5] = float("nan")
        x[1, 4097] = 2.0
        x[1, 8191:8193] = float("nan")
        x[1, -1] = -4.0
        x[2, 0] = float("inf")
        x[2, 1] = -float("inf")
        x[2, 8192] = float("nan")
        x[2, -1] = 1.0
        with self._inner_tree_flag():
            result = torch.nansum(x, dim=1)
        self.assertEqual(result[0].view(torch.int32).item(), 0)
        self.assertEqual(result[1].item(), -1.0)
        self.assertTrue(torch.isnan(result[2]).item())

    def test_nansum_unsupported_calls_fall_through(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_nansum_input(4, 32, torch.float32)
        complex_x = torch.ones(4, 32, device="cuda", dtype=torch.complex64)
        complex_x[:, ::17] = complex(float("nan"), 0.0)
        cases = [
            ("dim_none", x, {"dim": None}),
            ("multi_dim", x.reshape(2, 2, 32), {"dim": (1, 2)}),
            ("explicit_dtype", x, {"dim": 1, "dtype": torch.float64}),
            ("noncontiguous", x[:, ::2], {"dim": 1}),
            (
                "integer",
                torch.arange(128, device="cuda", dtype=torch.int64).reshape(4, 32),
                {"dim": 1},
            ),
            ("complex", complex_x, {"dim": 1}),
        ]
        for name, input_, kwargs in cases:
            with self.subTest(name=name):
                with torch.backends.python_native.cutedsl.disabled():
                    ref = torch.nansum(input_, **kwargs)
                with (
                    mock.patch.object(
                        inner_tree_kernel,
                        "inner_tree_nansum_into",
                        wraps=inner_tree_kernel.inner_tree_nansum_into,
                    ) as nansum_into,
                    self._inner_tree_flag(),
                ):
                    got = torch.nansum(input_, **kwargs)
                self.assertEqual(nansum_into.call_count, 0)
                self.assertEqual(got, ref, rtol=0, atol=0)

    # --- mean (sum DAG followed by one final division) ---

    @parametrize("dtype", [torch.float32, torch.float64])
    def test_mean_matches_sum_over_n(self, dtype):
        view_dtype = torch.int32 if dtype == torch.float32 else torch.int64
        for m, n in [(128, 15), (128, 8195), (8, 200003)]:
            with self.subTest(m=m, n=n):
                x = self._make_order_sensitive_input(m, n, dtype)
                with self._inner_tree_flag():
                    got = torch.mean(x, dim=1)
                    # IEEE true division (tensor/tensor, div.rn) to match the
                    # kernel's divide-by-N. NOTE: `tensor / python_int` lowers to
                    # reciprocal-multiply (sum * (1/n)) which differs by ~1 ULP.
                    expected = torch.sum(x, dim=1) / torch.tensor(
                        n, device=x.device, dtype=dtype
                    )
                self.assertTrue(
                    torch.equal(got.view(view_dtype), expected.view(view_dtype))
                )

    @parametrize("dtype", [torch.float32, torch.float64])
    def test_mean_matches_aten(self, dtype):
        rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-12, 1e-12)
        for m, n in [(128, 15), (128, 8195), (8, 200003)]:
            with self.subTest(m=m, n=n):
                x = self._make_order_sensitive_input(m, n, dtype)
                with torch.backends.python_native.cutedsl.disabled():
                    ref = torch.mean(x, dim=1)
                with self._inner_tree_flag():
                    got = torch.mean(x, dim=1)
                self.assertEqual(got, ref, rtol=rtol, atol=atol)

    def test_mean_override_engaged(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(128, 8195, torch.float32)
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_mean_into",
                wraps=inner_tree_kernel.inner_tree_mean_into,
            ) as mean_into,
            self._inner_tree_flag(),
        ):
            torch.mean(x, dim=1)
        self.assertEqual(mean_into.call_count, 1)

    def test_mean_override_out_variant(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(128, 8195, torch.float32)
        with self._inner_tree_flag():
            functional = torch.mean(x, dim=1)
        out = torch.empty(128, device="cuda", dtype=torch.float32)
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_mean_into",
                wraps=inner_tree_kernel.inner_tree_mean_into,
            ) as mean_into,
            self._inner_tree_flag(),
        ):
            returned = torch.mean(x, dim=1, out=out)
        self.assertEqual(mean_into.call_count, 1)
        self.assertIs(returned, out)
        self.assertTrue(
            torch.equal(out.view(torch.int32), functional.view(torch.int32))
        )

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mean_low_precision_matches_aten(self, dtype):
        from torch._native.ops.reductions import inner_tree_kernel

        for m, n in [(64, 32), (8, 4096), (8, 65536)]:
            with self.subTest(m=m, n=n):
                x = self._make_order_sensitive_input(m, n, dtype)
                with torch.backends.python_native.cutedsl.disabled():
                    ref = torch.mean(x, dim=1)
                with (
                    mock.patch.object(
                        inner_tree_kernel,
                        "inner_tree_mean_into",
                        wraps=inner_tree_kernel.inner_tree_mean_into,
                    ) as mean_into,
                    self._inner_tree_flag(),
                ):
                    got = torch.mean(x, dim=1)
                self.assertEqual(mean_into.call_count, 1)
                self.assertEqual(got, ref, rtol=2e-2, atol=2e-2)

    def test_mean_unsupported_calls_fall_through(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(4, 32, torch.float32)
        cases = [
            ("bare", x, {}),
            ("explicit_dtype", x, {"dim": 1, "dtype": torch.float64}),
            ("multi_dim", x.reshape(2, 2, 32), {"dim": (1, 2)}),
            ("noncontiguous", x[:, ::2], {"dim": 1}),
            ("complex", x.to(torch.complex64), {"dim": 1}),
        ]
        for name, input_, kwargs in cases:
            with self.subTest(name=name):
                with torch.backends.python_native.cutedsl.disabled():
                    ref = torch.mean(input_, **kwargs)
                with (
                    mock.patch.object(
                        inner_tree_kernel,
                        "inner_tree_mean_into",
                        wraps=inner_tree_kernel.inner_tree_mean_into,
                    ) as mean_into,
                    self._inner_tree_flag(),
                ):
                    got = torch.mean(input_, **kwargs)
                self.assertEqual(mean_into.call_count, 0)
                self.assertEqual(got, ref, rtol=0, atol=0)

        integer = torch.ones(4, 32, device="cuda", dtype=torch.int64)
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_mean_into",
                wraps=inner_tree_kernel.inner_tree_mean_into,
            ) as mean_into,
            self._inner_tree_flag(),
            self.assertRaisesRegex(RuntimeError, "could not infer output dtype"),
        ):
            torch.mean(integer, dim=1)
        self.assertEqual(mean_into.call_count, 0)

    # --- vector norm (abs/square pre-map plus optional sqrt post-map) ---

    def test_vector_norm_cond_orders(self):
        impl = _cutedsl_impl()
        x = self._make_order_sensitive_input(128, 8195, torch.float32)
        out = torch.empty(128, device="cuda", dtype=torch.float32)
        with self._inner_tree_flag():
            for order in (1, 1.0, 2, 2.0):
                with self.subTest(order=order):
                    self.assertTrue(impl._vector_norm_cond(x, order, [1]))
                    self.assertTrue(impl._vector_norm_out_cond(x, order, [1], out=out))
            self.assertTrue(impl._vector_norm_cond(x, dim=[1]))
            self.assertTrue(impl._vector_norm_out_cond(x, dim=[1], out=out))

            for order in (0, 3, float("inf"), -float("inf"), 1 + 0j, True):
                with self.subTest(rejected_order=order):
                    self.assertFalse(impl._vector_norm_cond(x, order, [1]))
                    self.assertFalse(impl._vector_norm_out_cond(x, order, [1], out=out))

            self.assertFalse(impl._vector_norm_cond(x, 2, [1], dtype=torch.float32))
            self.assertFalse(
                impl._vector_norm_out_cond(x, 2, [1], dtype=torch.float32, out=out)
            )
            self.assertFalse(impl._vector_norm_cond(x, 2, None))
            self.assertFalse(impl._vector_norm_out_cond(x, 2, None, out=out))
            x_3d = x.reshape(2, 64, 8195)
            self.assertFalse(impl._vector_norm_cond(x_3d, 2, [1, 2]))
            self.assertFalse(
                impl._vector_norm_out_cond(
                    x_3d,
                    2,
                    [1, 2],
                    out=torch.empty(2, device="cuda", dtype=torch.float32),
                )
            )

            for unsupported in (
                torch.ones(4, 32, device="cuda", dtype=torch.int64),
                torch.ones(4, 32, device="cuda", dtype=torch.complex64),
                x[:, ::2],
            ):
                with self.subTest(dtype=unsupported.dtype, stride=unsupported.stride()):
                    unsupported_out = torch.empty(
                        unsupported.shape[0],
                        device="cuda",
                        dtype=(
                            torch.float32
                            if unsupported.is_complex()
                            else unsupported.dtype
                        ),
                    )
                    self.assertFalse(impl._vector_norm_cond(unsupported, 2, [1]))
                    self.assertFalse(
                        impl._vector_norm_out_cond(
                            unsupported, 2, [1], out=unsupported_out
                        )
                    )

            singleton = torch.ones(4, 1, device="cuda", dtype=torch.float32)
            singleton_out = torch.empty(4, device="cuda", dtype=torch.float32)
            self.assertFalse(impl._vector_norm_cond(singleton, 2, [1]))
            self.assertFalse(
                impl._vector_norm_out_cond(singleton, 2, [1], out=singleton_out)
            )

            self.assertTrue(impl._vector_norm_cond(x, 2, [1]))
            for invalid_out in (
                torch.empty(129, device="cuda", dtype=torch.float32),
                torch.empty(128, device="cuda", dtype=torch.float64),
            ):
                self.assertFalse(impl._vector_norm_out_cond(x, 2, [1], out=invalid_out))

    @parametrize("dtype", [torch.float32, torch.float64])
    def test_vector_norm_matches_inner_tree_composition(self, dtype):
        view_dtype = torch.int32 if dtype == torch.float32 else torch.int64
        for order in (2, 1):
            for m, n in [(128, 15), (128, 8195), (8, 200003)]:
                with self.subTest(order=order, m=m, n=n):
                    x = self._make_order_sensitive_input(m, n, dtype)
                    with self._inner_tree_flag():
                        got = torch.linalg.vector_norm(x, ord=order, dim=1)
                        if order == 2:
                            expected = torch.sqrt(torch.sum(x * x, dim=1))
                        else:
                            expected = torch.sum(torch.abs(x), dim=1)
                    self.assertTrue(
                        torch.equal(got.view(view_dtype), expected.view(view_dtype))
                    )

    @parametrize("dtype", [torch.float32, torch.float64])
    def test_vector_norm_matches_aten(self, dtype):
        rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-12, 1e-12)
        for order in (2, 1):
            for m, n in [(128, 15), (128, 8195), (8, 200003)]:
                with self.subTest(order=order, m=m, n=n):
                    x = self._make_order_sensitive_input(m, n, dtype)
                    with torch.backends.python_native.cutedsl.disabled():
                        ref = torch.linalg.vector_norm(x, ord=order, dim=1)
                    with self._inner_tree_flag():
                        got = torch.linalg.vector_norm(x, ord=order, dim=1)
                    self.assertEqual(got, ref, rtol=rtol, atol=atol)

    def test_vector_norm_override_engaged(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(128, 8195, torch.float32)
        calls = [
            (
                "torch.linalg.vector_norm",
                lambda: torch.linalg.vector_norm(x, ord=2, dim=1),
            ),
            (
                "torch.linalg.norm",
                lambda: torch.linalg.norm(x, ord=2, dim=1),
            ),
            ("torch.norm", lambda: torch.norm(x, p=2, dim=1)),
            (
                "default_order",
                lambda: torch.linalg.vector_norm(x, dim=1),
            ),
        ]
        for name, call in calls:
            with self.subTest(name=name):
                with (
                    mock.patch.object(
                        inner_tree_kernel,
                        "inner_tree_vector_norm_into",
                        wraps=inner_tree_kernel.inner_tree_vector_norm_into,
                    ) as vector_norm_into,
                    self._inner_tree_flag(),
                ):
                    call()
                self.assertEqual(vector_norm_into.call_count, 1)

    def test_vector_norm_out_variant(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(128, 8195, torch.float32)
        for order in (2, 1):
            for keepdim in (False, True):
                with self.subTest(order=order, keepdim=keepdim):
                    with self._inner_tree_flag():
                        functional = torch.linalg.vector_norm(
                            x, ord=order, dim=1, keepdim=keepdim
                        )
                    out_shape = (128, 1) if keepdim else (128,)
                    out = torch.empty(out_shape, device="cuda", dtype=torch.float32)
                    with (
                        mock.patch.object(
                            inner_tree_kernel,
                            "inner_tree_vector_norm_into",
                            wraps=inner_tree_kernel.inner_tree_vector_norm_into,
                        ) as vector_norm_into,
                        self._inner_tree_flag(),
                    ):
                        returned = torch.linalg.vector_norm(
                            x, ord=order, dim=1, keepdim=keepdim, out=out
                        )
                    self.assertEqual(vector_norm_into.call_count, 1)
                    self.assertIs(returned, out)
                    self.assertTrue(
                        torch.equal(out.view(torch.int32), functional.view(torch.int32))
                    )

        with torch.backends.python_native.cutedsl.disabled():
            ref = torch.linalg.vector_norm(x, ord=3, dim=1)
        out = torch.empty(128, device="cuda", dtype=torch.float32)
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_vector_norm_into",
                wraps=inner_tree_kernel.inner_tree_vector_norm_into,
            ) as vector_norm_into,
            self._inner_tree_flag(),
        ):
            returned = torch.linalg.vector_norm(x, ord=3, dim=1, out=out)
        self.assertEqual(vector_norm_into.call_count, 0)
        self.assertIs(returned, out)
        self.assertEqual(out, ref, rtol=0, atol=0)

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_vector_norm_low_precision_matches_aten(self, dtype):
        from torch._native.ops.reductions import inner_tree_kernel

        for order in (2, 1):
            for m, n in [(64, 32), (8, 4096), (8, 65536)]:
                with self.subTest(order=order, m=m, n=n):
                    x = self._make_order_sensitive_input(m, n, dtype)
                    with torch.backends.python_native.cutedsl.disabled():
                        ref = torch.linalg.vector_norm(x, ord=order, dim=1)
                    with (
                        mock.patch.object(
                            inner_tree_kernel,
                            "inner_tree_vector_norm_into",
                            wraps=inner_tree_kernel.inner_tree_vector_norm_into,
                        ) as vector_norm_into,
                        self._inner_tree_flag(),
                    ):
                        got = torch.linalg.vector_norm(x, ord=order, dim=1)
                    self.assertEqual(vector_norm_into.call_count, 1)
                    self.assertEqual(got, ref, rtol=2e-2, atol=2e-2)

        overflow = torch.zeros(64, 32, device="cuda", dtype=dtype)
        overflow[:, :2] = 30000
        cases = [("overflow", overflow)]
        if dtype == torch.bfloat16:
            underflow = torch.zeros(64, 32, device="cuda", dtype=dtype)
            underflow[:, :2] = 2**-70
            cases.append(("underflow", underflow))

        for name, x in cases:
            with self.subTest(name=name):
                with torch.backends.python_native.cutedsl.disabled():
                    expected = torch.sqrt(
                        torch.sum(x.to(torch.float32) ** 2, dim=1)
                    ).to(dtype)
                with (
                    mock.patch.object(
                        inner_tree_kernel,
                        "inner_tree_vector_norm_into",
                        wraps=inner_tree_kernel.inner_tree_vector_norm_into,
                    ) as vector_norm_into,
                    self._inner_tree_flag(),
                ):
                    got = torch.linalg.vector_norm(x, ord=2, dim=1)
                self.assertEqual(vector_norm_into.call_count, 1)
                self.assertTrue(torch.isfinite(got).all().item())
                self.assertTrue(
                    torch.equal(got.view(torch.int16), expected.view(torch.int16))
                )

    def test_vector_norm_unsupported_calls_fall_through(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(4, 32, torch.float32)
        cases = [
            ("order_zero", x, {"ord": 0, "dim": 1}),
            ("order_three", x, {"ord": 3, "dim": 1}),
            ("order_inf", x, {"ord": float("inf"), "dim": 1}),
            ("order_neg_inf", x, {"ord": -float("inf"), "dim": 1}),
            ("explicit_dtype", x, {"ord": 2, "dim": 1, "dtype": torch.float64}),
            ("dim_none", x, {"ord": 2, "dim": None}),
            (
                "multi_dim",
                x.reshape(2, 2, 32),
                {"ord": 2, "dim": (1, 2)},
            ),
            ("noncontiguous", x[:, ::2], {"ord": 2, "dim": 1}),
            ("complex", x.to(torch.complex64), {"ord": 2, "dim": 1}),
        ]
        for name, input_, kwargs in cases:
            with self.subTest(name=name):
                with torch.backends.python_native.cutedsl.disabled():
                    ref = torch.linalg.vector_norm(input_, **kwargs)
                with (
                    mock.patch.object(
                        inner_tree_kernel,
                        "inner_tree_vector_norm_into",
                        wraps=inner_tree_kernel.inner_tree_vector_norm_into,
                    ) as vector_norm_into,
                    self._inner_tree_flag(),
                ):
                    got = torch.linalg.vector_norm(input_, **kwargs)
                self.assertEqual(vector_norm_into.call_count, 0)
                self.assertEqual(got, ref, rtol=0, atol=0)

        error_cases = [
            (
                "integer_input",
                torch.ones(4, 32, device="cuda", dtype=torch.int64),
                2,
                "floating point or complex",
            ),
            ("complex_order", x, 1 + 0j, "Expected a non-complex scalar"),
        ]
        for name, input_, order, error in error_cases:
            with self.subTest(name=name):
                with (
                    mock.patch.object(
                        inner_tree_kernel,
                        "inner_tree_vector_norm_into",
                        wraps=inner_tree_kernel.inner_tree_vector_norm_into,
                    ) as vector_norm_into,
                    self._inner_tree_flag(),
                    self.assertRaisesRegex(RuntimeError, error),
                ):
                    torch.linalg.vector_norm(input_, ord=order, dim=1)
                self.assertEqual(vector_norm_into.call_count, 0)

        limit = torch.finfo(torch.float32).max
        extreme = torch.tensor(
            [[limit], [-limit], [limit / 2], [-limit / 2]],
            device="cuda",
            dtype=torch.float32,
        )
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_vector_norm_into",
                wraps=inner_tree_kernel.inner_tree_vector_norm_into,
            ) as vector_norm_into,
            self._inner_tree_flag(),
        ):
            got = torch.linalg.vector_norm(extreme, ord=2, dim=1)
        self.assertEqual(vector_norm_into.call_count, 0)
        self.assertTrue(torch.isfinite(got).all().item())
        self.assertTrue(
            torch.equal(
                got.view(torch.int32), extreme.abs().squeeze(1).view(torch.int32)
            )
        )

    # --- nanmean (ATen composite over nansum + count/div) ---

    def _make_nanmean_input(self, m, n, dtype):
        x = self._make_order_sensitive_input(m, n, dtype)
        cols = torch.arange(n, device="cuda").reshape(1, n)
        rows = torch.arange(m, device="cuda").reshape(m, 1)
        return x.masked_fill((cols + 3 * rows) % 13 == 0, float("nan"))

    @parametrize("dtype", [torch.float32, torch.float64])
    def test_nanmean_matches_nansum_over_count(self, dtype):
        view_dtype = torch.int32 if dtype == torch.float32 else torch.int64
        for m, n in [(128, 15), (128, 8195), (8, 200003)]:
            with self.subTest(m=m, n=n):
                x = self._make_nanmean_input(m, n, dtype)
                with self._inner_tree_flag():
                    got = torch.nanmean(x, 1)
                    count = (~torch.isnan(x)).sum(1)
                    expected = torch.nansum(x, 1) / count
                self.assertEqual(count.dtype, torch.int64)
                self.assertEqual(
                    got.view(view_dtype), expected.view(view_dtype), rtol=0, atol=0
                )

    def test_nanmean_engaged(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_nanmean_input(128, 8195, torch.float32)
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_nansum_into",
                wraps=inner_tree_kernel.inner_tree_nansum_into,
            ) as nansum_into,
            self._inner_tree_flag(),
        ):
            torch.nanmean(x, dim=1)
        self.assertEqual(nansum_into.call_count, 1)

    @parametrize("dtype", [torch.float32, torch.float64])
    def test_nanmean_matches_aten(self, dtype):
        rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-12, 1e-12)
        for m, n in [(128, 15), (128, 8195), (8, 200003)]:
            with self.subTest(m=m, n=n):
                x = self._make_nanmean_input(m, n, dtype)
                x[0].fill_(float("nan"))
                with torch.backends.python_native.cutedsl.disabled():
                    ref = torch.nanmean(x, dim=1)
                with self._inner_tree_flag():
                    got = torch.nanmean(x, dim=1)
                self.assertTrue(torch.isnan(got[0]).item())
                self.assertEqual(got[1:], ref[1:], rtol=rtol, atol=atol)

    def test_nanmean_out_variant(self):
        x = self._make_nanmean_input(128, 8195, torch.float32)
        with self._inner_tree_flag():
            functional = torch.nanmean(x, dim=1)
        out = torch.empty(128, device="cuda", dtype=torch.float32)
        with self._inner_tree_flag():
            returned = torch.nanmean(x, dim=1, out=out)
        self.assertIs(returned, out)
        self.assertEqual(
            out.view(torch.int32), functional.view(torch.int32), rtol=0, atol=0
        )

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_nanmean_low_precision_matches_aten(self, dtype):
        from torch._native.ops.reductions import inner_tree_kernel

        for m, n in [(128, 15), (128, 8195), (8, 200003)]:
            with self.subTest(m=m, n=n):
                x = self._make_nanmean_input(m, n, dtype)
                with torch.backends.python_native.cutedsl.disabled():
                    ref = torch.nanmean(x, dim=1)
                with (
                    mock.patch.object(
                        inner_tree_kernel,
                        "inner_tree_nansum_into",
                        wraps=inner_tree_kernel.inner_tree_nansum_into,
                    ) as nansum_into,
                    self._inner_tree_flag(),
                ):
                    got = torch.nanmean(x, dim=1)
                self.assertEqual(nansum_into.call_count, 1)
                self.assertEqual(got, ref, rtol=2e-2, atol=2e-2)

    def test_nanmean_unsupported_calls_fall_through(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_nanmean_input(4, 32, torch.float32)
        cases = [
            ("dim_none", x, {"dim": None}),
            ("explicit_dtype", x, {"dim": 1, "dtype": torch.float64}),
            ("multi_dim", x.reshape(2, 2, 32), {"dim": (1, 2)}),
            ("noncontiguous", x[:, ::2], {"dim": 1}),
            ("complex", x.to(torch.complex64), {"dim": 1}),
        ]
        for name, input_, kwargs in cases:
            with self.subTest(name=name):
                with torch.backends.python_native.cutedsl.disabled():
                    ref = torch.nanmean(input_, **kwargs)
                with (
                    mock.patch.object(
                        inner_tree_kernel,
                        "inner_tree_nansum_into",
                        wraps=inner_tree_kernel.inner_tree_nansum_into,
                    ) as nansum_into,
                    self._inner_tree_flag(),
                ):
                    got = torch.nanmean(input_, **kwargs)
                self.assertEqual(nansum_into.call_count, 0)
                self.assertEqual(got, ref, rtol=0, atol=0)

        for dtype in (torch.int64, torch.bool):
            with self.subTest(dtype=dtype):
                integer = torch.ones(4, 32, device="cuda", dtype=dtype)
                with (
                    mock.patch.object(
                        inner_tree_kernel,
                        "inner_tree_nansum_into",
                        wraps=inner_tree_kernel.inner_tree_nansum_into,
                    ) as nansum_into,
                    self._inner_tree_flag(),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "expected input to have floating point or complex dtype",
                    ),
                ):
                    torch.nanmean(integer, dim=1)
                self.assertEqual(nansum_into.call_count, 0)

    def test_nanmean_requires_grad_is_warning_free(self):
        x = self._make_nanmean_input(4, 32, torch.float32).requires_grad_()
        with self._inner_tree_flag():
            self.assertNotWarn(lambda: torch.nanmean(x, dim=1).sum().backward())
        self.assertIsNotNone(x.grad)

    # --- logsumexp (ATen max masking/exp/log over inner-tree sum) ---

    @parametrize("dtype", [torch.float32, torch.float64])
    @parametrize("m,n", [(128, 15), (128, 8195), (8, 200003)])
    def test_logsumexp_matches_inner_tree_composition(self, dtype, m, n):
        view_dtype = torch.int32 if dtype == torch.float32 else torch.int64
        x = self._make_order_sensitive_input(m, n, dtype)
        with self._inner_tree_flag():
            got = torch.logsumexp(x, dim=1)
            maxes = torch.amax(x, dim=1, keepdim=True)
            squeezed_maxes = maxes.squeeze(1)
            safe = torch.where(torch.isfinite(maxes), maxes, 0)
            expected = squeezed_maxes + torch.log(torch.sum(torch.exp(x - safe), dim=1))
        self.assertEqual(
            got.view(view_dtype), expected.view(view_dtype), rtol=0, atol=0
        )

    @parametrize("m,n", [(128, 15), (128, 8195), (8, 200003)])
    def test_logsumexp_override_engaged(self, m, n):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(m, n, torch.float32)
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_sum_into",
                wraps=inner_tree_kernel.inner_tree_sum_into,
            ) as sum_into,
            self._inner_tree_flag(),
        ):
            torch.logsumexp(x, dim=1)
        self.assertEqual(sum_into.call_count, 1)

    @parametrize("dtype", [torch.float32, torch.float64])
    @parametrize("m,n", [(128, 15), (128, 8195), (8, 200003)])
    def test_logsumexp_matches_aten(self, dtype, m, n):
        rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-12, 1e-12)
        x = self._make_order_sensitive_input(m, n, dtype)
        with torch.backends.python_native.cutedsl.disabled():
            ref = torch.logsumexp(x, dim=1)
        with self._inner_tree_flag():
            got = torch.logsumexp(x, dim=1)
        self.assertEqual(got, ref, rtol=rtol, atol=atol)

    @parametrize("keepdim", [False, True])
    def test_logsumexp_out_variant(self, keepdim):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(128, 8195, torch.float32)
        with self._inner_tree_flag():
            functional = torch.logsumexp(x, dim=1, keepdim=keepdim)
        out_shape = (128, 1) if keepdim else (128,)
        out = torch.empty(out_shape, device="cuda", dtype=torch.float32)
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_sum_into",
                wraps=inner_tree_kernel.inner_tree_sum_into,
            ) as sum_into,
            self._inner_tree_flag(),
        ):
            returned = torch.logsumexp(x, dim=1, keepdim=keepdim, out=out)
        self.assertEqual(sum_into.call_count, 1)
        self.assertIs(returned, out)
        self.assertEqual(
            out.view(torch.int32), functional.view(torch.int32), rtol=0, atol=0
        )

    def test_logsumexp_mis_sized_out(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(128, 8195, torch.float32)
        with (
            torch.backends.python_native.cutedsl.disabled(),
            self.assertWarnsRegex(
                UserWarning, "An output with one or more elements was resized"
            ),
        ):
            native_out = torch.empty(1, device="cuda", dtype=torch.float32)
            native_returned = torch.logsumexp(x, dim=1, out=native_out)
        out = torch.empty(1, device="cuda", dtype=torch.float32)
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_sum_into",
                wraps=inner_tree_kernel.inner_tree_sum_into,
            ) as sum_into,
            self._inner_tree_flag(),
            self.assertWarnsRegex(
                UserWarning, "An output with one or more elements was resized"
            ),
        ):
            returned = torch.logsumexp(x, dim=1, out=out)
        self.assertEqual(sum_into.call_count, 0)
        self.assertIs(native_returned, native_out)
        self.assertIs(returned, out)
        self.assertEqual(out.shape, native_out.shape)
        self.assertEqual(
            out.view(torch.int32), native_out.view(torch.int32), rtol=0, atol=0
        )

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("m,n", [(64, 32), (8, 4096), (8, 65536)])
    def test_logsumexp_low_precision_matches_aten(self, dtype, m, n):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(m, n, dtype)
        with torch.backends.python_native.cutedsl.disabled():
            ref = torch.logsumexp(x, dim=1)
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_sum_into",
                wraps=inner_tree_kernel.inner_tree_sum_into,
            ) as sum_into,
            self._inner_tree_flag(),
        ):
            got = torch.logsumexp(x, dim=1)
        self.assertEqual(sum_into.call_count, 1)
        self.assertEqual(got, ref, rtol=2e-2, atol=2e-2)

    def test_logsumexp_all_negative_infinity(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(2, 8195, torch.float32)
        x[0].fill_(-float("inf"))
        with torch.backends.python_native.cutedsl.disabled():
            ref = torch.logsumexp(x, dim=1)
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_sum_into",
                wraps=inner_tree_kernel.inner_tree_sum_into,
            ) as sum_into,
            self._inner_tree_flag(),
        ):
            got = torch.logsumexp(x, dim=1)
        self.assertEqual(sum_into.call_count, 1)
        self.assertTrue(torch.isneginf(got[0]).item())
        self.assertFalse(torch.isnan(got[0]).item())
        self.assertEqual(got, ref, rtol=1e-5, atol=1e-6)

    def test_logsumexp_multi_dim_falls_through(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(4, 32, torch.float32).reshape(2, 2, 32)
        with torch.backends.python_native.cutedsl.disabled():
            ref = torch.logsumexp(x, dim=[0, 1])
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_sum_into",
                wraps=inner_tree_kernel.inner_tree_sum_into,
            ) as sum_into,
            self._inner_tree_flag(),
        ):
            got = torch.logsumexp(x, dim=[0, 1])
        self.assertEqual(sum_into.call_count, 0)
        self.assertEqual(got, ref, rtol=0, atol=0)

    @parametrize(
        "kwargs,error",
        [
            ({"dim": None}, "argument 'dim' must be tuple of ints"),
            (
                {"dim": 1, "dtype": torch.float64},
                "unexpected keyword argument 'dtype'",
            ),
        ],
    )
    def test_logsumexp_schema_errors(self, kwargs, error):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(4, 32, torch.float32)
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_sum_into",
                wraps=inner_tree_kernel.inner_tree_sum_into,
            ) as sum_into,
            self._inner_tree_flag(),
            self.assertRaisesRegex(TypeError, error),
        ):
            torch.logsumexp(x, **kwargs)
        self.assertEqual(sum_into.call_count, 0)

    def test_logsumexp_requires_grad(self):
        from torch._native.ops.reductions import inner_tree_kernel

        x = self._make_order_sensitive_input(4, 32, torch.float32).requires_grad_()
        with (
            mock.patch.object(
                inner_tree_kernel,
                "inner_tree_sum_into",
                wraps=inner_tree_kernel.inner_tree_sum_into,
            ) as sum_into,
            self._inner_tree_flag(),
        ):
            self.assertNotWarn(lambda: torch.logsumexp(x, dim=1).sum().backward())
        self.assertEqual(sum_into.call_count, 1)
        self.assertIsNotNone(x.grad)


instantiate_parametrized_tests(TestSumCuteDSLOverride)

if __name__ == "__main__":
    run_tests()
