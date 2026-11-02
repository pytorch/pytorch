# Owner(s): ["module: sdpa"]

import itertools
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock

import torch
import torch.nn.attention as attention
import torch.nn.functional as F
from torch.nn.attention import _cudnn, sdpa_kernel, SDPBackend
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_CUDNN_ATTENTION
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


_SHAPE = (2, 8, 256, 64)
_NO_CUDNN_ATTENTION = "cuDNN attention unsupported"
_PROVIDER = _cudnn._CUDNN_MODULE_PATH


def _cudnn_python_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if not torch.backends.cuda.is_cudnn_sdp_python_available():
        return False
    # Only a missing dependency means unavailable: any other failure to import
    # an installed provider must fail the tests rather than skip them.
    try:
        import cudnn.torch  # noqa: F401
    except ImportError:
        return False
    try:
        _cudnn._check_frontend_version(_PROVIDER)
    except RuntimeError:
        return False
    q = torch.empty(_SHAPE, device="cuda", dtype=torch.bfloat16)
    params = torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, True, False)
    return torch.backends.cuda.can_use_cudnn_attention(params, False)


def _varlen_cudnn_supported(device) -> bool:
    # varlen_attn selects cuDNN only on SM90 and SM100.
    return torch.cuda.get_device_capability(device)[0] in (9, 10)


def _varlen_inputs(device):
    torch.manual_seed(0)
    lens = torch.tensor([128, 256], device=device)
    cu_seq = F.pad(lens.cumsum(0), (1, 0)).to(torch.int32)
    total, max_len = int(cu_seq[-1]), int(lens.max())
    q, k, v = (
        torch.randn(
            total, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    )
    return q, k, v, cu_seq, max_len


def _fake_install(live, tag):
    """An install_sdpa_overrides whose installed kernels show up in ``live``."""

    class _Handle:
        def remove(self):
            live.remove(tag)

    def install():
        live.append(tag)
        return _Handle()

    return install


def _counting_install(calls):
    """An install_sdpa_overrides that, like cudnn.torch's, overrides the cuDNN SDPA op.

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

    def install():
        # Not _scoped_library: the switch owns this registration's lifetime,
        # and removing it through the returned handle is what is under test.
        lib = torch.library.Library("aten", "IMPL")  # noqa: SCOPED_LIBRARY
        lib.impl("_scaled_dot_product_cudnn_attention", forward, "CUDA")
        return Handle(lib)

    return install


def _use_provider(test, install, version=_cudnn._MIN_FRONTEND_VERSION):
    """Serve ``cudnn.torch`` from a stand-in module, switch off, until ``test`` ends.

    The stand-in's parent package ``cudnn`` reports ``version``.
    """
    package = types.ModuleType("cudnn")
    package.__version__ = version
    module = types.ModuleType(_PROVIDER)
    if install is not None:
        setattr(module, _cudnn._INSTALL_FN, install)
    patcher = mock.patch.dict(sys.modules, {"cudnn": package, _PROVIDER: module})
    patcher.start()
    test.addCleanup(patcher.stop)
    saved = _cudnn._ACTIVE_HANDLE
    _cudnn._ACTIVE_HANDLE = None

    def restore():
        _cudnn.disable()
        _cudnn._ACTIVE_HANDLE = saved

    test.addCleanup(restore)


class TestCuDNNPythonSDPALifecycle(TestCase):
    # Calls torch.nn.attention._cudnn directly: the torch.backends.cuda wrappers
    # only forward to it and are covered by TestCuDNNPythonSDPA, and GENERIC
    # classes may not call torch.backends.cuda.* (test linter: accelerator_api).
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        self.live = []

    def test_missing_provider_is_reported_and_not_enabled(self):
        # A None entry in sys.modules is how a missing package looks to import.
        _use_provider(self, None)
        with mock.patch.dict(sys.modules, {_PROVIDER: None}):
            self.assertFalse(_cudnn.is_available())
            with self.assertRaises(ImportError):
                _cudnn.enable()
        self.assertFalse(_cudnn.is_enabled())
        _cudnn.disable()

    def test_old_provider_names_the_required_version(self):
        # 1.32 has install_sdpa_overrides(), but its varlen overrides ignore the
        # backend varlen_attn selected.
        _use_provider(self, _fake_install(self.live, "cudnn"), version="1.32.1")
        required = f"nvidia-cudnn-frontend>={_cudnn._MIN_FRONTEND_VERSION}"
        with self.assertRaisesRegex(RuntimeError, f"{required}.*1.32.1"):
            _cudnn.enable()
        self.assertFalse(_cudnn.is_enabled())
        self.assertEqual(self.live, [])

    def test_install_without_a_handle_is_rejected(self):
        _use_provider(self, lambda: None)
        with self.assertRaisesRegex(TypeError, "no remove"):
            _cudnn.enable()
        self.assertFalse(_cudnn.is_enabled())

    def test_enable_then_disable(self):
        _use_provider(self, _fake_install(self.live, "cudnn"))
        _cudnn.enable()
        self.assertTrue(_cudnn.is_enabled())
        self.assertEqual(self.live, ["cudnn"])
        _cudnn.disable()
        self.assertFalse(_cudnn.is_enabled())
        self.assertEqual(self.live, [])
        _cudnn.disable()

    def test_enable_is_idempotent(self):
        _use_provider(self, _fake_install(self.live, "cudnn"))
        _cudnn.enable()
        _cudnn.enable()
        self.assertEqual(self.live, ["cudnn"])
        _cudnn.disable()
        self.assertEqual(self.live, [])

    def test_independent_of_the_flash_attention_registry(self):
        _use_provider(self, _fake_install(self.live, "cudnn"))
        flash = _fake_install(self.live, "flash")
        attention.register_flash_attention_impl("_TEST_FLASH", register_fn=flash)
        self.addCleanup(attention._registry._FLASH_ATTENTION_IMPLS.pop, "_TEST_FLASH")
        self.addCleanup(
            lambda: attention.current_flash_attention_impl()
            and attention.restore_flash_attention_impl()
        )

        _cudnn.enable()
        attention.activate_flash_attention_impl("_TEST_FLASH")
        self.assertTrue(_cudnn.is_enabled())
        self.assertEqual(sorted(self.live), ["cudnn", "flash"])
        attention.restore_flash_attention_impl()
        self.assertEqual(self.live, ["cudnn"])
        _cudnn.disable()
        self.assertEqual(self.live, [])

    def test_env_var(self):
        _use_provider(self, _fake_install(self.live, "cudnn"))
        for value, expected in (("", []), ("0", []), ("true", []), ("1", ["cudnn"])):
            with mock.patch.dict(os.environ, {_cudnn.ENV_VAR: value}):
                _cudnn._enable_from_env()
            self.assertEqual(self.live, expected, msg=f"value={value!r}")
            _cudnn.disable()

    def test_env_var_failure_warns_instead_of_raising(self):
        def broken():
            raise RuntimeError("provider is broken")

        _use_provider(self, broken)
        with (
            mock.patch.dict(os.environ, {_cudnn.ENV_VAR: "1"}),
            self.assertLogs(_cudnn.logger, level="WARNING"),
        ):
            _cudnn._enable_from_env()
        self.assertFalse(_cudnn.is_enabled())

    def test_env_var_enables_at_import(self):
        # Runs the hook at the end of torch/__init__.py in a fresh interpreter,
        # with a stand-in package on the path in place of nvidia-cudnn-frontend.
        # The hook needs a fully initialised `torch`: placed earlier in
        # `import torch` (e.g. in torch/nn/attention/__init__.py), it fails.
        provider = textwrap.dedent(
            f"""
            import torch, torch.fx, torch.jit  # what the real provider imports

            class _Handle:
                def remove(self):
                    pass

            def {_cudnn._INSTALL_FN}():
                return _Handle()
            """
        )
        probe = "import torch; print(torch.backends.cuda.cudnn_sdp_python_enabled())"
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "cudnn", "torch"))
            with open(os.path.join(root, "cudnn", "__init__.py"), "w") as f:
                f.write(f"__version__ = {_cudnn._MIN_FRONTEND_VERSION!r}\n")
            with open(os.path.join(root, "cudnn", "torch", "__init__.py"), "w") as f:
                f.write(provider)
            path = os.pathsep.join(filter(None, (root, os.environ.get("PYTHONPATH"))))
            for value, expected in (("1", "True"), ("", "False")):
                env = dict(os.environ, PYTHONPATH=path, **{_cudnn.ENV_VAR: value})
                cmd = [sys.executable, "-c", probe]
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                self.assertEqual(
                    proc.stdout.split()[-1], expected, msg=f"value={value!r}"
                )
                self.assertNotIn("could not be enabled", proc.stderr)


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
        calls = []
        _use_provider(self, _counting_install(calls))
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
    def test_varlen_follows_the_selected_backend(self, device):
        # The provider serves the varlen calls that selected cuDNN and leaves
        # the ones that selected flash to torch.
        import cudnn.torch as provider

        from torch.nn.attention.varlen import varlen_attn

        if not _varlen_cudnn_supported(device):
            self.skipTest("varlen_attn selects cuDNN only on SM90 and SM100")
        q, k, v, cu_seq, max_len = _varlen_inputs(device)
        grad = torch.randn_like(q)

        def run(backend):
            with sdpa_kernel(backend):
                out = varlen_attn(
                    q, k, v, cu_seq, cu_seq, max_len, max_len, window_size=(-1, 0)
                )
            return (out, *torch.autograd.grad(out, (q, k, v), grad))

        for backend, served in (
            (SDPBackend.CUDNN_ATTENTION, 1),
            (SDPBackend.FLASH_ATTENTION, 0),
        ):
            with self.subTest(backend=backend):
                expected = run(backend)
                torch.backends.cuda.enable_cudnn_sdp_python(True)
                fwd0, bwd0 = provider.calls["fwd"], provider.calls["bwd"]
                got = run(backend)
                fwd, bwd = provider.calls["fwd"] - fwd0, provider.calls["bwd"] - bwd0
                torch.backends.cuda.enable_cudnn_sdp_python(False)
                self.assertEqual((fwd, bwd), (served, served))
                for g, e in zip(got, expected):
                    self.assertEqual(g, e, atol=3e-2, rtol=3e-2)

    @unittest.skipUnless(_cudnn_python_available(), "cuDNN Python SDPA unavailable")
    def test_varlen_out_follows_the_selected_backend(self, device):
        import cudnn.torch as provider

        from torch.nn.attention.varlen import varlen_attn, varlen_attn_out

        if not _varlen_cudnn_supported(device):
            self.skipTest("varlen_attn selects cuDNN only on SM90 and SM100")
        q, k, v, cu_seq, max_len = _varlen_inputs(device)
        args = (q, k, v, cu_seq, cu_seq, max_len, max_len)
        torch.backends.cuda.enable_cudnn_sdp_python(True)
        for backend, served in (
            (SDPBackend.CUDNN_ATTENTION, 1),
            (SDPBackend.FLASH_ATTENTION, 0),
        ):
            with self.subTest(backend=backend), torch.no_grad(), sdpa_kernel(backend):
                out = torch.empty_like(q)
                fwd0 = provider.calls["fwd"]
                varlen_attn_out(out, *args, window_size=(-1, 0))
                self.assertEqual(provider.calls["fwd"] - fwd0, served)
                expected = varlen_attn(*args, window_size=(-1, 0))
                self.assertEqual(out, expected, atol=3e-2, rtol=3e-2)

    @unittest.skipUnless(_cudnn_python_available(), "cuDNN Python SDPA unavailable")
    @parametrize("layout", ["bhsd", "bshd"])
    @parametrize("is_causal", [False, True])
    def test_compiled_matches_eager(self, device, layout, is_causal):
        # The provider replaces the CUDA kernels but not the ATen meta
        # functions, which tracing uses; Inductor checks the provider's outputs
        # against the traced shapes and strides, forward and backward.
        import cudnn.torch as provider

        torch.manual_seed(0)
        b, h, s, d = _SHAPE
        shape = (b, h, s, d) if layout == "bhsd" else (b, s, h, d)

        def make():
            t = torch.randn(shape, device=device, dtype=torch.bfloat16)
            return (t if layout == "bhsd" else t.transpose(1, 2)).requires_grad_(True)

        q, k, v = make(), make(), make()
        grad = torch.randn(_SHAPE, device=device, dtype=torch.bfloat16)

        def sdpa(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        torch.backends.cuda.enable_cudnn_sdp_python(True)
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            eager = sdpa(q, k, v)
        eager = (eager, *torch.autograd.grad(eager, (q, k, v), grad))

        fwd0, bwd0 = provider.calls["fwd"], provider.calls["bwd"]
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            compiled = torch.compile(sdpa, fullgraph=True)(q, k, v)
        compiled = (compiled, *torch.autograd.grad(compiled, (q, k, v), grad))
        fwd, bwd = provider.calls["fwd"] - fwd0, provider.calls["bwd"] - bwd0
        self.assertEqual((fwd, bwd), (1, 1))
        for got, want in zip(compiled, eager):
            self.assertEqual(got.stride(), want.stride())
            self.assertEqual(got, want, atol=2e-2, rtol=2e-2)

    @unittest.skipUnless(_cudnn_python_available(), "cuDNN Python SDPA unavailable")
    def test_outputs_match_the_aten_meta_function(self, device):
        # Every query layout: the provider's outputs must have the shapes,
        # strides and dtypes that the retained ATen meta function reports.
        import cudnn.torch as provider

        op = torch.ops.aten._scaled_dot_product_cudnn_attention
        torch.backends.cuda.enable_cudnn_sdp_python(True)
        for order in itertools.permutations(range(3)):
            with self.subTest(order=order):
                dims = [_SHAPE[i] for i in order] + [_SHAPE[3]]
                inverse = [order.index(i) for i in range(3)] + [3]
                q, k, v = (
                    torch.randn(dims, device=device, dtype=torch.bfloat16).permute(
                        inverse
                    )
                    for _ in range(3)
                )
                meta_qkv = (
                    torch.empty_strided(
                        t.shape, t.stride(), dtype=t.dtype, device="meta"
                    )
                    for t in (q, k, v)
                )
                fwd0 = provider.calls["fwd"]
                got = op(q, k, v, None, True, is_causal=True)
                self.assertEqual(provider.calls["fwd"], fwd0 + 1)
                want = op(*meta_qkv, None, True, is_causal=True)
                for i, (g, w) in enumerate(zip(got, want)):
                    if isinstance(g, torch.Tensor):
                        self.assertEqual(
                            (g.shape, g.stride(), g.dtype),
                            (w.shape, w.stride(), w.dtype),
                            msg=f"output {i}",
                        )

    @unittest.skipUnless(_cudnn_python_available(), "cuDNN Python SDPA unavailable")
    def test_env_var_enables_the_real_provider_at_import(self, device):
        probe = "import torch; print(torch.backends.cuda.cudnn_sdp_python_enabled())"
        env = dict(os.environ, **{_cudnn.ENV_VAR: "1"})
        cmd = [sys.executable, "-c", probe]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(proc.stdout.split()[-1], "True")
        self.assertNotIn("could not be enabled", proc.stderr)


instantiate_device_type_tests(TestCuDNNPythonSDPA, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
