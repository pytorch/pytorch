# Owner(s): ["module: dsl-native-ops"]

import unittest

import torch
import torch._native
import torch.backends.python_native as pn
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


_REGISTER_KS = (2, 4, 8, 16)
_RADIX_KS = (64, 128, 256, 512, 1024)
_SUPPORTED_KS = _REGISTER_KS + _RADIX_KS


def _flydsl_topk_registered() -> bool:
    try:
        ops = pn.get_dsl_operations("flydsl")
        return "topk" in ops and "topk.values" in ops
    except Exception:
        return False


def _test_n(k: int) -> int:
    from torch._native.ops.topk.flydsl_impl import _RADIX_N_RANGE, _REGISTER_N_RANGE

    if k in _REGISTER_KS:
        return _REGISTER_N_RANGE[0]
    return _RADIX_N_RANGE[k][0]


@unittest.skipUnless(TEST_CUDA and torch.version.hip is not None, "ROCm required")
@unittest.skipUnless(_flydsl_topk_registered(), "FlyDSL topk override not registered")
class TestFlyDSLTopK(TestCase):
    def setUp(self):
        super().setUp()
        from torch._native.ops.topk.flydsl_kernels import clear_topk_cache

        clear_topk_cache()

    def _make_input(self, *, shape=(512, 512)):
        torch.manual_seed(0)
        return torch.randn(shape, device="cuda", dtype=torch.float32)

    def _assert_topk_matches_aten(self, x: torch.Tensor, k: int) -> None:
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        got_v, got_i = torch.topk(x, k, dim=-1)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)
        if k >= 2:
            diffs = got_v[..., :-1] - got_v[..., 1:]
            self.assertTrue((diffs >= 0).all(), "output is not descending")

    @parametrize("k", _SUPPORTED_KS)
    def test_correctness_random_gaussian(self, k: int):
        torch.manual_seed(0)
        x = torch.randn(256, _test_n(k), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", _SUPPORTED_KS)
    def test_correctness_with_duplicates(self, k: int):
        torch.manual_seed(1)
        x = torch.randint(0, 50, (256, _test_n(k)), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", _SUPPORTED_KS)
    def test_correctness_with_extreme_values(self, k: int):
        torch.manual_seed(2)
        x = torch.randn(256, _test_n(k), device="cuda", dtype=torch.float32)
        x[:, 0] = float("inf")
        x[:, 1] = float("-inf")
        x[:, 2] = 1e38
        x[:, 3] = -1e38
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", (8, 512))
    def test_correctness_with_nan(self, k: int):
        import struct

        torch.manual_seed(10)
        x = torch.randn(256, _test_n(k), device="cuda", dtype=torch.float32)
        neg_nan = struct.unpack("<f", struct.pack("<I", 0xFFC00000))[0]
        x[:, 0] = float("nan")
        x[:, 1] = neg_nan
        x[:, 2] = float("inf")
        x[:, 3] = float("-inf")
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        got_v, got_i = torch.topk(x, k, dim=-1)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)
        self.assertEqual(got_v.isnan().sum(dim=-1), ref_v.isnan().sum(dim=-1))
        ref_finite = ref_v.masked_select(~ref_v.isnan()).reshape(256, -1)
        got_finite = got_v.masked_select(~got_v.isnan()).reshape(256, -1)
        self.assertEqual(got_finite, ref_finite)

    @parametrize("k", _SUPPORTED_KS)
    def test_nd_input(self, k: int):
        torch.manual_seed(3)
        n = _test_n(k)
        x = torch.randn(4, 64, n, device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)
        got_v, _ = torch.topk(x, k, dim=-1)
        self.assertEqual(got_v.shape, (4, 64, k))

    @parametrize("k", _SUPPORTED_KS)
    def test_out_variant(self, k: int):
        torch.manual_seed(4)
        x = torch.randn(256, _test_n(k), device="cuda", dtype=torch.float32)
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)

        out_v = torch.empty(256, k, dtype=torch.float32, device="cuda")
        out_i = torch.empty(256, k, dtype=torch.int64, device="cuda")
        got_v, got_i = torch.topk(x, k, dim=-1, out=(out_v, out_i))
        self.assertIs(got_v, out_v)
        self.assertIs(got_i, out_i)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)

    def test_topk_uses_cache(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        # Use a (k, n) that hits the register-kernel gate so the FlyDSL path
        # actually compiles; both calls share one specialization so the second
        # is a cache hit.
        k = 8
        n = _test_n(k)
        x = self._make_input(shape=(256, n))
        with pn.flydsl.disabled():
            ref = torch.topk(x, k, dim=-1)

        got = torch.topk(x, k, dim=-1)
        self.assertEqual(got.values, ref.values)
        self.assertEqual(got.indices, ref.indices)

        x3 = self._make_input(shape=(8, 32, n))
        with pn.flydsl.disabled():
            ref3 = torch.topk(x3, k, dim=-1)
        got3 = torch.topk(x3, k, dim=-1)
        self.assertEqual(got3.values, ref3.values)
        self.assertEqual(got3.indices, ref3.indices)

        info = topk_cache_info()
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 1)
        self.assertEqual(info.currsize, 1)

    def test_user_disable_falls_back_to_aten(self):
        from torch._native.ops.topk.flydsl_kernels import topk_cache_info

        x = self._make_input()
        with pn.flydsl.disabled():
            ref = torch.topk(x, 8, dim=-1)
            got = torch.topk(x, 8, dim=-1)

        self.assertEqual(got.values, ref.values)
        self.assertEqual(got.indices, ref.indices)
        self.assertEqual(topk_cache_info().misses, 0)

    def test_unsupported_k_falls_through(self):
        torch.manual_seed(5)
        x = torch.randn(256, 512, device="cuda", dtype=torch.float32)
        for bad_k in (1, 10, 32):
            with pn.flydsl.disabled():
                ref = torch.topk(x, bad_k, dim=-1)
            got = torch.topk(x, bad_k, dim=-1)
            self.assertEqual(got.values, ref.values)
            self.assertEqual(got.indices, ref.indices)

    def test_register_non_power_of_two_n_falls_through(self):
        torch.manual_seed(8)
        x = torch.randn(256, 768, device="cuda", dtype=torch.float32)
        with pn.flydsl.disabled():
            ref = torch.topk(x, 8, dim=-1)
        got = torch.topk(x, 8, dim=-1)
        self.assertEqual(got.values, ref.values)
        self.assertEqual(got.indices, ref.indices)

    def test_radix_non_multiple_of_four_n(self):
        torch.manual_seed(11)
        x = torch.randn(256, 16385, device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, 512)

    @parametrize("k", (64, 512))
    def test_deterministic_mode_matches_aten_with_heavy_ties(self, k: int):
        torch.manual_seed(6)
        x = torch.randint(0, 4, (256, _test_n(k)), device="cuda", dtype=torch.float32)
        with pn.flydsl.disabled():
            ref_v, ref_i = torch.topk(x, k, dim=-1)

        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            v1, i1 = torch.topk(x, k, dim=-1)
            v2, i2 = torch.topk(x, k, dim=-1)
        finally:
            torch.use_deterministic_algorithms(prior)
        self.assertEqual(v1, v2)
        self.assertEqual(i1, i2)
        self.assertEqual(v1, ref_v)
        self.assertEqual(i1, ref_i)

    @parametrize("k", _REGISTER_KS)
    def test_register_stable_with_heavy_ties(self, k: int):
        torch.manual_seed(9)
        x = torch.randint(0, 4, (256, _test_n(k)), device="cuda", dtype=torch.float32)
        with pn.flydsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        v1, i1 = torch.topk(x, k, dim=-1)
        v2, i2 = torch.topk(x, k, dim=-1)
        self.assertEqual(v1, v2)
        self.assertEqual(i1, i2)
        self.assertEqual(v1, ref_v)
        self.assertEqual(torch.gather(x, -1, i1), v1)

    def test_autograd_passes_through(self):
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


instantiate_parametrized_tests(TestFlyDSLTopK)


if __name__ == "__main__":
    run_tests()
