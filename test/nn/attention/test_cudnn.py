# Owner(s): ["module: sdpa"]

import os
import subprocess
import sys
import unittest
from unittest import mock

import torch
import torch.nn.attention as attention
import torch.nn.functional as F
from torch.nn.attention import _cudnn, _registry, sdpa_kernel, SDPBackend
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_CUDNN_ATTENTION
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


_SHAPE = (2, 8, 256, 64)
_NO_CUDNN_ATTENTION = "cuDNN attention unsupported"


def _cudnn_python_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if not torch.backends.cuda.is_cudnn_sdp_python_available():
        return False
    q = torch.empty(_SHAPE, device="cuda", dtype=torch.bfloat16)
    params = torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, True, False)
    return torch.backends.cuda.can_use_cudnn_attention(params, False)


def _fake_impl(live, tag):
    """A register_fn whose installed kernels show up in ``live``."""

    class _Handle:
        def remove(self):
            live.remove(tag)

    def register():
        live.append(tag)
        return _Handle()

    return register


def _counting_cudnn_provider(calls):
    """A register_fn that, like cudnn.torch, overrides the cuDNN SDPA op.

    The override counts its calls and forwards to the built-in worker op, so it
    exercises the dispatcher without nvidia-cudnn-frontend.
    """

    def forward(q, k, v, bias, lse, dropout_p=0.0, causal=False, debug=False, **kw):
        calls.append(1)
        max_q, max_k = q.size(-2), k.size(-2)
        args = (q, k, v, bias, None, None, max_q, max_k, lse, dropout_p, causal, debug)
        return torch.ops.aten._cudnn_attention_forward(*args, **kw)

    class Handle:
        def __init__(self, lib):
            self.lib = lib

        def remove(self):
            # Dropping the last reference would also deregister, but only once
            # it is collected.
            self.lib._destroy()

    def register():
        lib = torch.library.Library("aten", "IMPL")
        lib.impl("_scaled_dot_product_cudnn_attention", forward, "CUDA")
        return Handle(lib)

    return register


def _isolate_switch_state(test):
    """Empty the flash registry and turn the switch off until ``test`` ends."""
    impls = _registry._FLASH_ATTENTION_IMPLS
    saved = (
        dict(impls),
        _registry._FLASH_ATTENTION_ACTIVE,
        _cudnn._PROVIDER_REGISTER_FN,
        _cudnn._ACTIVE_HANDLE,
    )

    def restore():
        impls.clear()
        impls.update(saved[0])
        _registry._FLASH_ATTENTION_ACTIVE = saved[1]
        _cudnn._PROVIDER_REGISTER_FN, _cudnn._ACTIVE_HANDLE = saved[2:]

    test.addCleanup(restore)
    impls.clear()
    _registry._FLASH_ATTENTION_ACTIVE = None
    _cudnn._PROVIDER_REGISTER_FN = _cudnn._ACTIVE_HANDLE = None


class TestCuDNNPythonSDPALifecycle(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        _isolate_switch_state(self)
        self.live = []

    def _register_provider(self):
        # What importing the provider does; "types" then stands in for it.
        register = _fake_impl(self.live, "cudnn")
        attention.register_flash_attention_impl("CUDNN", register_fn=register)

    def test_missing_provider_is_reported_and_not_enabled(self):
        # A None entry in sys.modules is how a missing package looks to import.
        with mock.patch.dict(sys.modules, {"cudnn.torch": None}):
            self.assertFalse(torch.backends.cuda.is_cudnn_sdp_python_available())
            with self.assertRaises(ImportError):
                torch.backends.cuda.enable_cudnn_sdp_python(True)
        self.assertFalse(torch.backends.cuda.cudnn_sdp_python_enabled())
        torch.backends.cuda.enable_cudnn_sdp_python(False)

    def test_provider_without_registration_raises(self):
        with self.assertRaisesRegex(RuntimeError, "did not register"):
            _cudnn.enable("types")
        self.assertFalse(_cudnn.is_enabled())

    def test_enable_takes_provider_out_of_flash_registry(self):
        self._register_provider()
        _cudnn.enable("types")
        self.assertEqual(self.live, ["cudnn"])
        self.assertNotIn("CUDNN", attention.list_flash_attention_impls())

    def test_activating_a_flash_impl_leaves_cudnn_enabled(self):
        self._register_provider()
        _cudnn.enable("types")
        flash = _fake_impl(self.live, "flash")
        attention.register_flash_attention_impl("FLASH", register_fn=flash)
        attention.activate_flash_attention_impl("FLASH")
        self.assertTrue(_cudnn.is_enabled())
        self.assertEqual(sorted(self.live), ["cudnn", "flash"])

    def test_enable_adopts_a_registry_activated_provider(self):
        self._register_provider()
        attention.activate_flash_attention_impl("CUDNN")
        handle = _registry._FLASH_ATTENTION_ACTIVE[1]
        _cudnn.enable("types")
        # Adopted, not reinstalled: there is no window in which a failing
        # reinstall could leave cuDNN off.
        self.assertIs(_cudnn._ACTIVE_HANDLE, handle)
        self.assertEqual(self.live, ["cudnn"])
        self.assertIsNone(attention.current_flash_attention_impl())

    def test_registry_activated_provider_is_reported_enabled(self):
        self._register_provider()
        attention.activate_flash_attention_impl("CUDNN")
        self.assertTrue(torch.backends.cuda.cudnn_sdp_python_enabled())
        self.assertIsNone(attention.current_flash_attention_impl())

    def test_disable_removes_a_registry_activated_provider(self):
        self._register_provider()
        attention.activate_flash_attention_impl("CUDNN")
        torch.backends.cuda.enable_cudnn_sdp_python(False)
        self.assertEqual(self.live, [])
        self.assertFalse(torch.backends.cuda.cudnn_sdp_python_enabled())
        self.assertIsNone(attention.current_flash_attention_impl())

    def test_enable_is_idempotent(self):
        self._register_provider()
        _cudnn.enable("types")
        _cudnn.enable("types")
        self.assertEqual(self.live, ["cudnn"])
        _cudnn.disable()
        self.assertEqual(self.live, [])

    def test_env_var(self):
        register = _fake_impl(self.live, "cudnn")
        for value, expected in (("", []), ("0", []), ("1", ["cudnn"])):
            with (
                mock.patch.object(_cudnn, "_PROVIDER_REGISTER_FN", register),
                mock.patch.dict(os.environ, {_cudnn.ENV_VAR: value}),
            ):
                _cudnn._enable_from_env()
            self.assertEqual(self.live, expected, msg=f"value={value!r}")

    def test_env_var_failure_warns_instead_of_raising(self):
        def broken():
            raise RuntimeError("provider is broken")

        with (
            mock.patch.object(_cudnn, "_PROVIDER_REGISTER_FN", broken),
            mock.patch.dict(os.environ, {_cudnn.ENV_VAR: "1"}),
            self.assertLogs(_cudnn.logger, level="WARNING"),
        ):
            _cudnn._enable_from_env()
        self.assertFalse(_cudnn.is_enabled())


class TestCuDNNPythonSDPA(TestCase):
    hw_classification = HardwareClassification.CUDA

    def tearDown(self):
        torch.backends.cuda.enable_cudnn_sdp_python(False)
        super().tearDown()

    def _sdpa(self, q, backend=SDPBackend.CUDNN_ATTENTION):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, q, q, is_causal=True)

    @unittest.skipUnless(PLATFORM_SUPPORTS_CUDNN_ATTENTION, _NO_CUDNN_ATTENTION)
    def test_switch_routes_the_cudnn_op(self, device):
        _isolate_switch_state(self)
        calls = []
        register = _counting_cudnn_provider(calls)
        attention.register_flash_attention_impl("CUDNN", register_fn=register)
        q = torch.randn(_SHAPE, device=device, dtype=torch.bfloat16)

        ref = self._sdpa(q)
        torch.backends.cuda.enable_cudnn_sdp_python(True)
        self.assertEqual(self._sdpa(q), ref)
        self.assertEqual(len(calls), 1)
        # sdpa_kernel(MATH) turns enable_cudnn_sdp off, which must still win.
        self._sdpa(q, SDPBackend.MATH)
        self.assertEqual(len(calls), 1)
        torch.backends.cuda.enable_cudnn_sdp_python(False)
        self.assertEqual(self._sdpa(q), ref)
        self.assertEqual(len(calls), 1)

    @unittest.skipUnless(PLATFORM_SUPPORTS_CUDNN_ATTENTION, _NO_CUDNN_ATTENTION)
    def test_switch_turns_off_a_registry_activated_provider(self, device):
        _isolate_switch_state(self)
        calls = []
        register = _counting_cudnn_provider(calls)
        attention.register_flash_attention_impl("CUDNN", register_fn=register)
        q = torch.randn(_SHAPE, device=device, dtype=torch.bfloat16)

        attention.activate_flash_attention_impl("CUDNN")
        self._sdpa(q)
        self.assertEqual(len(calls), 1)
        torch.backends.cuda.enable_cudnn_sdp_python(False)
        self._sdpa(q)
        self.assertEqual(len(calls), 1)

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
