# Owner(s): ["module: sdpa"]

import os
import subprocess
import sys
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from torch.nn.attention import _cudnn, SDPBackend, sdpa_kernel
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


_SHAPE = (2, 8, 256, 64)


def _cudnn_python_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if not torch.backends.cuda.is_cudnn_sdp_python_available():
        return False
    q = torch.empty(_SHAPE, device="cuda", dtype=torch.bfloat16)
    params = torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, True, False)
    return torch.backends.cuda.can_use_cudnn_attention(params, False)


class TestCuDNNPythonSDPAGating(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_missing_provider_is_reported_and_not_enabled(self):
        # A None entry in sys.modules is how a missing package looks to import.
        with (
            mock.patch.dict(sys.modules, {"cudnn.torch": None}),
            mock.patch.object(_cudnn, "_PROVIDER_REGISTER_FN", None),
        ):
            self.assertFalse(torch.backends.cuda.is_cudnn_sdp_python_available())
            with self.assertRaises(ImportError):
                torch.backends.cuda.enable_cudnn_sdp_python(True)
            self.assertFalse(torch.backends.cuda.cudnn_sdp_python_enabled())


class TestCuDNNPythonSDPA(TestCase):
    hw_classification = HardwareClassification.CUDA

    def tearDown(self):
        torch.backends.cuda.enable_cudnn_sdp_python(False)
        super().tearDown()

    @unittest.skipUnless(_cudnn_python_available(), "cuDNN Python SDPA unavailable")
    def test_selected_backend_serves_forward_and_backward(self, device):
        # Both implementations launch cuDNN kernels, so kernel names cannot tell
        # them apart; the provider's dispatch counter can.
        import cudnn.torch as provider

        torch.manual_seed(0)
        q, k, v = (
            torch.randn(_SHAPE, device=device, dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        )
        grad = torch.randn(_SHAPE, device=device, dtype=torch.bfloat16)

        torch.backends.cuda.enable_cudnn_sdp_python(True)
        calls = provider.calls
        fwd0, bwd0 = calls["fwd"], calls["bwd"]
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.backward(grad)
        self.assertEqual((calls["fwd"], calls["bwd"]), (fwd0 + 1, bwd0 + 1))

        qr, kr, vr = (t.detach().float().requires_grad_(True) for t in (q, k, v))
        with sdpa_kernel(SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True)
        ref.backward(grad.float())
        pairs = ((out, ref), (q.grad, qr.grad), (k.grad, kr.grad), (v.grad, vr.grad))
        for got, want in pairs:
            self.assertEqual(got.float(), want, atol=3e-2, rtol=3e-2)

    @unittest.skipUnless(_cudnn_python_available(), "cuDNN Python SDPA unavailable")
    def test_disable_restores_builtin_path(self, device):
        import cudnn.torch as provider

        q = torch.randn(_SHAPE, device=device, dtype=torch.bfloat16)
        torch.backends.cuda.enable_cudnn_sdp_python(True)
        torch.backends.cuda.enable_cudnn_sdp_python(False)
        self.assertFalse(torch.backends.cuda.cudnn_sdp_python_enabled())

        fwd0 = provider.calls["fwd"]
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            F.scaled_dot_product_attention(q, q, q, is_causal=True)
        self.assertEqual(provider.calls["fwd"], fwd0)

    @unittest.skipUnless(_cudnn_python_available(), "cuDNN Python SDPA unavailable")
    def test_env_var_enables_at_import(self, device):
        # The hook needs a fully initialised `torch`. It used to sit in
        # torch/nn/attention/__init__.py, which is imported partway through
        # `import torch`, and there it always failed.
        probe = "import torch; print(torch.backends.cuda.cudnn_sdp_python_enabled())"
        for value, expected in (("1", "True"), ("", "False")):
            env = dict(os.environ, TORCH_CUDNN_SDPA_USE_PYTHON=value)
            cmd = [sys.executable, "-c", probe]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertEqual(proc.stdout.split()[-1], expected, msg=f"value={value!r}")
            self.assertNotIn("could not be enabled", proc.stderr)


instantiate_device_type_tests(TestCuDNNPythonSDPA, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
