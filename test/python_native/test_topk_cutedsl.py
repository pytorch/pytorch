# Owner(s): ["module: dsl-native-ops"]
#
# Correctness tests for the CuTeDSL topk override (register + radix
# kernels).
#
# Radix kernel (K in {64,128,256,512,1024}) has two specialisations:
#   * Default (non-deterministic): atomic-counter gather + ord-only sort.
#     Returns correct top-K values; indices may differ from aten on ties
#     and may differ across successive calls. Tests verify values match
#     aten bit-exactly and ``gather(input, indices)`` reproduces values.
#   * Deterministic (under torch.use_deterministic_algorithms): prefix-sum
#     gather + lex ``(ord, -idx)`` sort. Bit-exact aten match on both
#     values and indices, stable across runs.
#
# Register kernel (K in {16,32}) is bit-exact on values and stable across
# runs. Index ordering on ties is ``(value desc, idx asc)``, which doesn't
# match aten's small-K CUDA topk; tests check value-equality + gather
# round-trip rather than index-equality.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


_RADIX_KS = (64, 128, 256, 512, 1024)
_REGISTER_KS = (16, 32)
_SUPPORTED_KS = _REGISTER_KS + _RADIX_KS


def _radix_min_n_for_k(k: int) -> int:
    """Mirror of the radix kernel's per-K N gate; used to size test
    shapes so the override actually fires rather than falling through
    to aten."""
    from torch._native.ops.topk.cutedsl_impl import _RADIX_MIN_N_MULTIPLIER

    return _RADIX_MIN_N_MULTIPLIER[k] * k


def _test_n(k: int) -> int:
    """Pick a test N for which the override fires."""
    if k in _REGISTER_KS:
        # Register kernel has a (min, max) N range per K. Pick max so it
        # exercises the largest in-range tile.
        from torch._native.ops.topk.cutedsl_impl import _REGISTER_N_RANGE

        return _REGISTER_N_RANGE[k][1]
    return max(_radix_min_n_for_k(k), 4096)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestCuTeDSLTopK(TestCase):
    def _assert_topk_matches_aten(self, x: torch.Tensor, k: int) -> None:
        """Compare the override (default non-deterministic mode) against
        aten. Values must match bit-exactly; indices may differ on ties
        but ``gather(input, indices)`` must still reproduce values."""
        pn = torch.backends.python_native
        with pn.cutedsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        got_v, got_i = torch.topk(x, k, dim=-1)
        self.assertEqual(got_v, ref_v)
        gathered = torch.gather(x, -1, got_i)
        self.assertEqual(gathered, got_v)
        # descending
        if k >= 2:
            diffs = got_v[..., :-1] - got_v[..., 1:]
            self.assertTrue((diffs >= 0).all(), "output is not descending")

    @parametrize("k", _SUPPORTED_KS)
    def test_correctness_random_gaussian(self, k: int) -> None:
        torch.manual_seed(0)
        # M chosen to exceed typical SM count so the override actually fires.
        x = torch.randn(256, _test_n(k), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", _SUPPORTED_KS)
    def test_correctness_with_duplicates(self, k: int) -> None:
        """Duplicates stress the threshold-bin gather path where
        non-deterministic index tie-breaking occurs."""
        torch.manual_seed(1)
        x = torch.randint(0, 50, (256, _test_n(k)), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", _SUPPORTED_KS)
    def test_correctness_with_extreme_values(self, k: int) -> None:
        torch.manual_seed(2)
        x = torch.randn(256, _test_n(k), device="cuda", dtype=torch.float32)
        x[:, 0] = float("inf")
        x[:, 1] = float("-inf")
        x[:, 2] = 1e38
        x[:, 3] = -1e38
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", _SUPPORTED_KS)
    def test_correctness_with_nan(self, k: int) -> None:
        """NaN of either sign must be classified as NaN (sorts at the top),
        matching aten's TopKTypeConfig<float> behaviour. The radix-ordinal
        encoder flags NaN explicitly so negative-sign NaN doesn't sort below
        -inf. We can't compare bit-exactly with aten because aten's final
        sort over the K gathered values uses fp32 ``<`` (NaN comparisons all
        false), so NaN positions within the top-K are unstable on aten's
        side. Instead we check the invariants: the same number of NaNs and
        the same set of finite values appear in our output and aten's."""
        import struct

        torch.manual_seed(10)
        x = torch.randn(256, _test_n(k), device="cuda", dtype=torch.float32)
        neg_nan = struct.unpack("<f", struct.pack("<I", 0xFFC00000))[0]
        x[:, 0] = float("nan")
        x[:, 1] = neg_nan
        x[:, 2] = float("inf")
        x[:, 3] = float("-inf")
        pn = torch.backends.python_native
        with pn.cutedsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        got_v, got_i = torch.topk(x, k, dim=-1)
        # gather round-trip: every output value must come from the input
        # at the reported index (works for NaN since assertEqual treats
        # NaN==NaN).
        self.assertEqual(torch.gather(x, -1, got_i), got_v)
        # Same NaN count per row as aten (proves negative-NaN didn't sink
        # below -inf).
        self.assertEqual(got_v.isnan().sum(dim=-1), ref_v.isnan().sum(dim=-1))
        # Finite tails must match aten bit-exactly (sorted descending).
        ref_finite = ref_v.masked_select(~ref_v.isnan()).reshape(256, -1)
        got_finite = got_v.masked_select(~got_v.isnan()).reshape(256, -1)
        self.assertEqual(got_finite, ref_finite)

    @parametrize("k", _SUPPORTED_KS)
    def test_nd_input(self, k: int) -> None:
        """Leading dims should be flattened to 2D; output should be
        reshaped back to self.shape[:-1] + (k,)."""
        torch.manual_seed(3)
        N = _test_n(k)
        x = torch.randn(4, 64, N, device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)
        got_v, _ = torch.topk(x, k, dim=-1)
        self.assertEqual(got_v.shape, (4, 64, k))

    @parametrize("k", _SUPPORTED_KS)
    def test_out_variant(self, k: int) -> None:
        torch.manual_seed(4)
        x = torch.randn(256, _test_n(k), device="cuda", dtype=torch.float32)
        pn = torch.backends.python_native
        with pn.cutedsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)

        out_v = torch.empty(256, k, dtype=torch.float32, device="cuda")
        out_i = torch.empty(256, k, dtype=torch.int64, device="cuda")
        got_v, got_i = torch.topk(x, k, dim=-1, out=(out_v, out_i))
        self.assertIs(got_v, out_v)
        self.assertIs(got_i, out_i)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)

    def test_unsupported_k_falls_through(self) -> None:
        """K values outside the supported set must hit aten and stay correct."""
        torch.manual_seed(5)
        x = torch.randn(256, 16384, device="cuda", dtype=torch.float32)
        pn = torch.backends.python_native
        for bad_k in (8, 100, 2048):
            with pn.cutedsl.disabled():
                ref = torch.topk(x, bad_k, dim=-1)
            got = torch.topk(x, bad_k, dim=-1)
            self.assertEqual(got.values, ref.values)
            self.assertEqual(got.indices, ref.indices)

    def test_register_n_out_of_range_falls_through(self) -> None:
        """Register K values with N outside the per-K cap should fall through."""
        torch.manual_seed(8)
        pn = torch.backends.python_native
        # K=32 register accepts N <= 256; N=512 must fall through (radix
        # doesn't accept K=32 either).
        x = torch.randn(256, 512, device="cuda", dtype=torch.float32)
        with pn.cutedsl.disabled():
            ref = torch.topk(x, 32, dim=-1)
        got = torch.topk(x, 32, dim=-1)
        self.assertEqual(got.values, ref.values)
        self.assertEqual(got.indices, ref.indices)

    @parametrize("k", _RADIX_KS)
    def test_deterministic_mode_matches_aten_with_heavy_ties(self, k: int) -> None:
        """Under ``torch.use_deterministic_algorithms`` the radix kernel
        uses prefix-sum gather + lex ``(ord, -idx)`` sort and must match
        aten bit-exactly on both values and indices, even with heavy ties."""
        torch.manual_seed(6)
        x = torch.randint(0, 4, (256, _test_n(k)), device="cuda", dtype=torch.float32)
        pn = torch.backends.python_native
        with pn.cutedsl.disabled():
            ref_v, ref_i = torch.topk(x, k, dim=-1)

        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            v1, i1 = torch.topk(x, k, dim=-1)
            # Determinism: two successive calls return identical results.
            v2, i2 = torch.topk(x, k, dim=-1)
        finally:
            torch.use_deterministic_algorithms(prior)
        self.assertEqual(v1, v2)
        self.assertEqual(i1, i2)
        # And bit-exact match to aten.
        self.assertEqual(v1, ref_v)
        self.assertEqual(i1, ref_i)

    @parametrize("k", _REGISTER_KS)
    def test_register_stable_with_heavy_ties(self, k: int) -> None:
        """Register kernel index order on ties is ``(value desc, idx asc)``,
        which doesn't match aten's small-K CUDA topk. Check the weaker
        contract: bit-exact values, gather round-trip, and stability
        across successive calls."""
        torch.manual_seed(9)
        x = torch.randint(0, 4, (256, _test_n(k)), device="cuda", dtype=torch.float32)
        pn = torch.backends.python_native
        with pn.cutedsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        v1, i1 = torch.topk(x, k, dim=-1)
        v2, i2 = torch.topk(x, k, dim=-1)
        self.assertEqual(v1, v2)
        self.assertEqual(i1, i2)
        self.assertEqual(v1, ref_v)
        self.assertEqual(torch.gather(x, -1, i1), v1)

    def test_autograd_passes_through(self) -> None:
        """aten's derivative handles backward; override runs at CUDA key
        below Autograd, so grad should still be the standard scatter-of-1s."""
        torch.manual_seed(7)
        x = torch.randn(
            256, 4096, device="cuda", dtype=torch.float32, requires_grad=True
        )
        v, i = torch.topk(x, 256, dim=-1)
        v.sum().backward()
        self.assertIsNotNone(x.grad)
        expected = torch.zeros_like(x)
        expected.scatter_(-1, i, 1.0)
        self.assertEqual(x.grad, expected)


instantiate_parametrized_tests(TestCuTeDSLTopK)


if __name__ == "__main__":
    run_tests()
