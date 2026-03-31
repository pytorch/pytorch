# Owner(s): ["module: dsl-native-ops"]

import unittest
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


_CUTEDSL_AVAILABLE = False
try:
    from torch._native import cutedsl_utils as _cutedsl_utils

    _CUTEDSL_AVAILABLE = _cutedsl_utils.runtime_available()
except Exception:
    pass

_IS_SM100 = False
if torch.cuda.is_available():
    try:
        major, _ = torch.cuda.get_device_capability()
        _IS_SM100 = major >= 10
    except Exception:
        pass


class TestOinkRmsnormGating(TestCase):
    """Test that the rmsnorm gating function correctly accepts/rejects inputs."""

    def _get_gating_fn(self):
        from torch._native.ops.oink_rmsnorm.rmsnorm import _should_use_oink_rmsnorm

        return _should_use_oink_rmsnorm

    def test_rejects_cpu_tensor(self):
        fn = self._get_gating_fn()
        x = torch.randn(4, 128, dtype=torch.bfloat16)
        self.assertFalse(fn(x))

    def test_rejects_wrong_dtype(self):
        fn = self._get_gating_fn()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        x = torch.randn(4, 128, dtype=torch.float32, device="cuda")
        self.assertFalse(fn(x))

    def test_rejects_non_2d(self):
        fn = self._get_gating_fn()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        x = torch.randn(2, 4, 128, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(fn(x))

    def test_rejects_non_tensor(self):
        fn = self._get_gating_fn()
        self.assertFalse(fn("not a tensor"))
        self.assertFalse(fn(None))

    def test_rejects_when_runtime_unavailable(self):
        fn = self._get_gating_fn()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
        with patch(
            "torch._native.ops.oink_rmsnorm.rmsnorm.cu.runtime_available",
            return_value=False,
        ):
            self.assertFalse(fn(x))

    def test_rejects_non_sm100(self):
        fn = self._get_gating_fn()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
        with patch(
            "torch.cuda.get_device_capability",
            return_value=(8, 0),
        ):
            self.assertFalse(fn(x))


class TestOinkRmsnormRegistration(TestCase):
    """Verify that rmsnorm is registered in the override graph."""

    def test_rmsnorm_registered(self):
        from torch._native.registry import _graphs

        self.assertIn(("_fused_rms_norm", "CUDA"), _graphs)


@unittest.skipIf(
    not (_IS_SM100 and _CUTEDSL_AVAILABLE),
    "Requires SM100 and CuTeDSL runtime",
)
class TestOinkRmsnormCorrectness(TestCase):
    """Correctness tests: compare oink rmsnorm against reference."""

    def test_rmsnorm_matches_reference(self):
        torch.manual_seed(42)
        M, N = 128, 1024
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        eps = 1e-5

        # Reference
        rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        ref = (x.float() * rms).to(x.dtype) * w

        # Via aten op (should dispatch to oink on SM100)
        result, rstd = torch.ops.aten._fused_rms_norm(x, [N], w, eps)

        torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)
        self.assertEqual(rstd.shape, (M,))

    def test_rmsnorm_various_shapes(self):
        torch.manual_seed(42)
        for M, N in [(1, 512), (32, 4096), (64, 7168), (256, 8192)]:
            x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
            eps = 1e-5

            rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
            ref = (x.float() * rms).to(x.dtype) * w

            result, rstd = torch.ops.aten._fused_rms_norm(x, [N], w, eps)
            torch.testing.assert_close(
                result,
                ref,
                atol=1e-2,
                rtol=1e-2,
                msg=f"Failed for shape ({M}, {N})",
            )


class TestOinkSoftmaxGating(TestCase):
    """Test that the softmax gating function correctly accepts/rejects inputs."""

    def _get_gating_fn(self):
        from torch._native.ops.oink_softmax.softmax import _should_use_oink_softmax

        return _should_use_oink_softmax

    def test_rejects_cpu_tensor(self):
        fn = self._get_gating_fn()
        x = torch.randn(4, 128, dtype=torch.bfloat16)
        self.assertFalse(fn(x, -1))

    def test_rejects_non_2d(self):
        fn = self._get_gating_fn()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        x = torch.randn(2, 4, 128, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(fn(x, -1))

    def test_rejects_wrong_dim(self):
        fn = self._get_gating_fn()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(fn(x, 0))


class TestOinkSoftmaxRegistration(TestCase):
    """Verify that softmax is registered in the override graph."""

    def test_softmax_registered(self):
        from torch._native.registry import _graphs

        self.assertIn(("_softmax.out", "CUDA"), _graphs)


@unittest.skipIf(
    not (_IS_SM100 and _CUTEDSL_AVAILABLE),
    "Requires SM100 and CuTeDSL runtime",
)
class TestOinkSoftmaxCorrectness(TestCase):
    """Correctness tests: compare oink softmax against reference."""

    def test_softmax_matches_reference(self):
        torch.manual_seed(42)
        x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
        ref = torch.softmax(x.float(), dim=-1).to(x.dtype)
        result = torch._softmax(x, -1, False)
        torch.testing.assert_close(result, ref, atol=1e-3, rtol=1e-3)

    def test_softmax_various_shapes(self):
        torch.manual_seed(42)
        for M, N in [(1, 256), (32, 1024), (128, 4096)]:
            x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
            ref = torch.softmax(x.float(), dim=-1).to(x.dtype)
            result = torch._softmax(x, -1, False)
            torch.testing.assert_close(
                result,
                ref,
                atol=1e-3,
                rtol=1e-3,
                msg=f"Failed for shape ({M}, {N})",
            )


if __name__ == "__main__":
    run_tests()
