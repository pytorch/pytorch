# Owner(s): ["module: dsl-native-ops"]

import unittest

import torch
import torch.backends.python_native as pn
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase


EPS = 1e-5


def _flydsl_rmsnorm_registered() -> bool:
    try:
        operations = set(pn.get_dsl_operations("flydsl"))
        return {
            "_fused_rms_norm",
            "_fused_rms_norm_backward",
        }.issubset(operations)
    except Exception:
        return False


def _tolerance(dtype: torch.dtype, result: str) -> tuple[float, float]:
    # These bounds follow the upstream FlyDSL RMSNorm tests. dweight uses fp32
    # atomics and is cast at the end, so its accumulation order needs a larger
    # absolute tolerance than output/rstd/dx.
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


def _assert_close(test_case, actual, expected, dtype, result):
    rtol, atol = _tolerance(dtype, result)
    test_case.assertEqual(actual.shape, expected.shape)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@unittest.skipUnless(TEST_CUDA and torch.version.hip is not None, "ROCm required")
@unittest.skipUnless(
    _flydsl_rmsnorm_registered(), "FlyDSL RMSNorm overrides are not registered"
)
class TestFlyDSLRMSNorm(TestCase):
    def setUp(self):
        super().setUp()
        from torch._native.ops.norm.flydsl_kernels import clear_rmsnorm_caches

        clear_rmsnorm_caches()
        torch.manual_seed(0)

    def _make_inputs(self, m, n, dtype, *, requires_grad=False):
        x = torch.randn(
            (m, n), device="cuda", dtype=dtype, requires_grad=requires_grad
        )
        weight = torch.randn(
            (n,), device="cuda", dtype=dtype, requires_grad=requires_grad
        )
        grad_out = torch.randn((m, n), device="cuda", dtype=dtype)
        return x, weight, grad_out

    def test_internal_forward_matches_aten_and_reuses_cache(self):
        from torch._native.ops.norm.flydsl_kernels import rmsnorm_cache_info

        x, weight, _ = self._make_inputs(16, 128, torch.float16)
        with pn.flydsl.disabled():
            ref_out, ref_rstd = torch.ops.aten._fused_rms_norm.default(
                x, [128], weight, EPS
            )

        out, rstd = torch.ops.aten._fused_rms_norm.default(
            x, [128], weight, EPS
        )
        self.assertEqual(rstd.dtype, torch.float32)
        self.assertEqual(rstd.shape, (16, 1))
        self.assertEqual(rstd.device, x.device)
        _assert_close(self, out, ref_out, x.dtype, "output")
        _assert_close(self, rstd, ref_rstd, x.dtype, "rstd")

        # A 3-D input with the same N/dtype/device must hit the same dynamic-M
        # specialization instead of compiling a second kernel.
        x3 = torch.randn((2, 16, 128), device="cuda", dtype=x.dtype)
        out3, _ = torch.ops.aten._fused_rms_norm.default(
            x3, [128], weight, EPS
        )
        self.assertEqual(out3.shape, x3.shape)

        info = rmsnorm_cache_info()["fwd"]
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 1)
        self.assertEqual(info.currsize, 1)

    def test_internal_backward_rezeros_atomic_buffer_and_honors_masks(self):
        from torch._native.ops.norm.flydsl_kernels import rmsnorm_cache_info

        x, weight, grad_out = self._make_inputs(16, 128, torch.float16)
        with pn.flydsl.disabled():
            _, rstd = torch.ops.aten._fused_rms_norm.default(
                x, [128], weight, EPS
            )
            ref_dx, ref_dw = (
                torch.ops.aten._fused_rms_norm_backward.default(
                    grad_out, x, [128], rstd, weight, [True, True]
                )
            )

        # The first launch catches compile-time execution that was not cleared;
        # the second protects against future accidental reuse of the atomic buffer.
        for _ in range(2):
            got_dx, got_dw = (
                torch.ops.aten._fused_rms_norm_backward.default(
                    grad_out, x, [128], rstd, weight, [True, True]
                )
            )
            _assert_close(self, got_dx, ref_dx, x.dtype, "dx")
            _assert_close(self, got_dw, ref_dw, x.dtype, "dweight")
            self.assertEqual(got_dx.dtype, x.dtype)
            self.assertEqual(got_dw.dtype, weight.dtype)

        dx_only, missing_dw = torch.ops.aten._fused_rms_norm_backward.default(
            grad_out, x, [128], rstd, weight, [True, False]
        )
        self.assertIsNone(missing_dw)
        _assert_close(self, dx_only, ref_dx, x.dtype, "dx")

        missing_dx, dw_only = torch.ops.aten._fused_rms_norm_backward.default(
            grad_out, x, [128], rstd, weight, [False, True]
        )
        self.assertIsNone(missing_dx)
        _assert_close(self, dw_only, ref_dw, x.dtype, "dweight")

        info = rmsnorm_cache_info()["bwd"]
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 3)

    def test_public_autograd_forward_and_backward_matrix(self):
        from torch._native.ops.norm.flydsl_kernels import rmsnorm_cache_info

        configs = (
            (64, 256, torch.float32),
            (16, 512, torch.bfloat16),
            (128, 4096, torch.float16),
        )
        for m, n, dtype in configs:
            with self.subTest(m=m, n=n, dtype=dtype):
                x, weight, grad_out = self._make_inputs(m, n, dtype)

                x_ref = x.detach().clone().requires_grad_(True)
                w_ref = weight.detach().clone().requires_grad_(True)
                with pn.flydsl.disabled():
                    ref_out = torch.rms_norm(x_ref, (n,), w_ref, EPS)
                    ref_dx, ref_dw = torch.autograd.grad(
                        ref_out, (x_ref, w_ref), grad_out
                    )

                x_got = x.detach().clone().requires_grad_(True)
                w_got = weight.detach().clone().requires_grad_(True)
                got_out = torch.rms_norm(x_got, (n,), w_got, EPS)
                got_dx, got_dw = torch.autograd.grad(
                    got_out, (x_got, w_got), grad_out
                )

                _assert_close(self, got_out, ref_out, dtype, "output")
                _assert_close(self, got_dx, ref_dx, dtype, "dx")
                _assert_close(self, got_dw, ref_dw, dtype, "dweight")

        info = rmsnorm_cache_info()
        self.assertEqual(info["fwd"].misses, len(configs))
        self.assertEqual(info["bwd"].misses, len(configs))

    def test_nn_rmsnorm_module_reaches_flydsl_backward(self):
        from torch._native.ops.norm.flydsl_kernels import rmsnorm_cache_info

        m, n, dtype = 16, 128, torch.float16
        x, weight, grad_out = self._make_inputs(m, n, dtype)

        ref_layer = torch.nn.RMSNorm(n, eps=EPS, device="cuda", dtype=dtype)
        got_layer = torch.nn.RMSNorm(n, eps=EPS, device="cuda", dtype=dtype)
        with torch.no_grad():
            ref_layer.weight.copy_(weight)
            got_layer.weight.copy_(weight)

        x_ref = x.detach().clone().requires_grad_(True)
        with pn.flydsl.disabled():
            ref_out = ref_layer(x_ref)
            ref_out.backward(grad_out)

        x_got = x.detach().clone().requires_grad_(True)
        got_out = got_layer(x_got)
        got_out.backward(grad_out)

        _assert_close(self, got_out, ref_out, dtype, "output")
        _assert_close(self, x_got.grad, x_ref.grad, dtype, "dx")
        _assert_close(
            self,
            got_layer.weight.grad,
            ref_layer.weight.grad,
            dtype,
            "dweight",
        )
        self.assertEqual(rmsnorm_cache_info()["bwd"].misses, 1)

    def test_unsupported_eps_falls_back_without_compiling(self):
        from torch._native.ops.norm.flydsl_kernels import rmsnorm_cache_info

        x, weight, _ = self._make_inputs(16, 128, torch.float16)
        unsupported_eps = 1e-6
        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (128,), weight, unsupported_eps)
        got = torch.rms_norm(x, (128,), weight, unsupported_eps)

        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 0)
        self.assertEqual(rmsnorm_cache_info()["bwd"].misses, 0)

    def test_noncontiguous_internal_input_falls_back(self):
        from torch._native.ops.norm.flydsl_kernels import rmsnorm_cache_info

        base = torch.randn((128, 16), device="cuda", dtype=torch.float16)
        x = base.transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        weight = torch.randn((128,), device="cuda", dtype=x.dtype)

        with pn.flydsl.disabled():
            ref, _ = torch.ops.aten._fused_rms_norm.default(
                x, [128], weight, EPS
            )
        got, _ = torch.ops.aten._fused_rms_norm.default(
            x, [128], weight, EPS
        )

        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 0)

    def test_misaligned_vector_input_falls_back(self):
        from torch._native.ops.norm.flydsl_kernels import rmsnorm_cache_info

        n = 4096
        storage = torch.randn((2 * n + 1,), device="cuda", dtype=torch.float16)
        weight_storage = torch.randn((n + 1,), device="cuda", dtype=torch.float16)
        x = storage[1:].view(2, n)
        weight = weight_storage[1:]
        self.assertTrue(x.is_contiguous())
        self.assertNotEqual(x.data_ptr() % 16, 0)
        self.assertNotEqual(weight.data_ptr() % 16, 0)

        with pn.flydsl.disabled():
            ref, _ = torch.ops.aten._fused_rms_norm.default(
                x, [n], weight, EPS
            )
        got, _ = torch.ops.aten._fused_rms_norm.default(x, [n], weight, EPS)

        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 0)

    def test_user_disable_falls_back_without_compiling(self):
        from torch._native.ops.norm.flydsl_kernels import rmsnorm_cache_info

        x, weight, _ = self._make_inputs(16, 128, torch.float16)
        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (128,), weight, EPS)
            got = torch.rms_norm(x, (128,), weight, EPS)

        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 0)


if __name__ == "__main__":
    run_tests()
