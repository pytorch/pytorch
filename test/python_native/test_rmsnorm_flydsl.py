# Owner(s): ["module: dsl-native-ops"]

import unittest

import torch
import torch.backends.python_native as pn
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase


def _flydsl_rmsnorm_registered() -> bool:
    try:
        return "rms_norm" in pn.get_dsl_operations("flydsl")
    except Exception:
        return False


@unittest.skipUnless(TEST_CUDA and torch.version.hip is not None, "ROCm required")
@unittest.skipUnless(_flydsl_rmsnorm_registered(), "FlyDSL RMSNorm override not registered")
class TestFlyDSLRMSNorm(TestCase):
    def setUp(self):
        super().setUp()
        from torch._native.ops.norm.flydsl_kernels import clear_rmsnorm_cache

        clear_rmsnorm_cache()

    def _make_inputs(self, *, eps=1e-5):
        torch.manual_seed(0)
        x = torch.randn((16, 128), device="cuda", dtype=torch.float16)
        weight = torch.randn((128,), device="cuda", dtype=torch.float16)
        return x, weight, eps

    def test_rms_norm_matches_aten_and_uses_cache(self):
        from torch._native.ops.norm.flydsl_kernels import rmsnorm_cache_info

        x, weight, eps = self._make_inputs()
        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (128,), weight, eps)

        got = torch.rms_norm(x, (128,), weight, eps)
        torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)

        x3 = torch.randn((2, 8, 128), device="cuda", dtype=torch.float16)
        with pn.flydsl.disabled():
            ref3 = torch.rms_norm(x3, (128,), weight, eps)
        got3 = torch.rms_norm(x3, (128,), weight, eps)
        torch.testing.assert_close(got3, ref3, atol=1e-2, rtol=1e-2)

        info = rmsnorm_cache_info()
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 1)
        self.assertEqual(info.currsize, 1)

    def test_user_disable_falls_back_to_aten(self):
        from torch._native.ops.norm.flydsl_kernels import rmsnorm_cache_info

        x, weight, eps = self._make_inputs()
        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (128,), weight, eps)
            got = torch.rms_norm(x, (128,), weight, eps)

        torch.testing.assert_close(got, ref, atol=0, rtol=0)
        self.assertEqual(rmsnorm_cache_info().misses, 0)

    def test_unsupported_eps_falls_back_to_aten(self):
        from torch._native.ops.norm.flydsl_kernels import rmsnorm_cache_info

        x, weight, _ = self._make_inputs()
        eps = 1e-6
        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (128,), weight, eps)

        got = torch.rms_norm(x, (128,), weight, eps)
        torch.testing.assert_close(got, ref, atol=0, rtol=0)
        self.assertEqual(rmsnorm_cache_info().misses, 0)


if __name__ == "__main__":
    run_tests()
