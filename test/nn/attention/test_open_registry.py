# Owner(s): ["module: sdpa"]

import os
from unittest import mock

import torch
import torch.nn.attention as attention
from torch.nn.attention import _cudnn, _registry
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class FakeHandle:
    def remove(self):
        pass


class TestFlashAttentionRegistry(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        self._saved_impls = dict(_registry._FLASH_ATTENTION_IMPLS)
        self._saved_active = attention.current_flash_attention_impl()
        _registry._FLASH_ATTENTION_IMPLS.clear()
        _registry._FLASH_ATTENTION_ACTIVE = None

    def tearDown(self):
        _registry._FLASH_ATTENTION_IMPLS.clear()
        _registry._FLASH_ATTENTION_IMPLS.update(self._saved_impls)
        _registry._FLASH_ATTENTION_ACTIVE = self._saved_active
        super().tearDown()

    def test_register_and_activate_impl(self):
        calls: dict[str, bool] = {}

        def fake_register():
            calls["called"] = True
            return FakeHandle()

        attention.register_flash_attention_impl("TEST_FA", register_fn=fake_register)
        self.assertIn("TEST_FA", attention.list_flash_attention_impls())

        attention.activate_flash_attention_impl("TEST_FA")

        self.assertTrue(calls.get("called", False))
        self.assertEqual("TEST_FA", attention.current_flash_attention_impl())

    def test_activate_unknown_impl_errors(self):
        with self.assertRaisesRegex(
            ValueError, "Unknown flash attention impl 'missing'"
        ):
            attention.activate_flash_attention_impl("missing")

    def test_cudnn_impl_is_registered(self):
        """Importing torch.nn.attention registers CUDNN -- the point of the
        in-tree shim. _saved_impls is the snapshot setUp took before clearing
        the registry, so this asserts what import time actually produced."""
        self.assertIn("CUDNN", self._saved_impls)

    def test_cudnn_missing_package_raises_and_keeps_default(self):
        """Without nvidia-cudnn-frontend installed, activation reports the
        missing module rather than leaving a half-registered state."""
        with self.assertRaises(ModuleNotFoundError):
            _cudnn.register_cudnn_attention("cudnn_frontend_not_installed_xyz")
        self.assertIsNone(attention.current_flash_attention_impl())

    def test_cudnn_package_without_registry_support_raises(self):
        """A provider too old to register itself must produce a clear error
        instead of recursing back into this shim: the registered callable is
        still the shim after the import, so calling it again would loop."""
        attention.register_flash_attention_impl(
            "CUDNN", register_fn=_cudnn.register_cudnn_attention
        )
        # types is always importable and registers nothing.
        with self.assertRaisesRegex(RuntimeError, "did not register"):
            _cudnn.register_cudnn_attention("types")

    def test_cudnn_python_coexists_with_flash_impl(self):
        """The backends.cuda switch and the flash registry are independent
        axes: they override disjoint operators, so turning on the Python cuDNN
        implementation must not disturb the active flash implementation."""
        installed = set()

        def _make(tag):
            class _H:
                def remove(self):
                    installed.discard(tag)

            def _register():
                installed.add(tag)
                return _H()

            return _register

        attention.register_flash_attention_impl("FA_FAKE", register_fn=_make("flash"))
        attention.activate_flash_attention_impl("FA_FAKE")

        self.addCleanup(_cudnn.disable)
        with mock.patch.object(_cudnn, "_PROVIDER_REGISTER_FN", _make("cudnn")):
            torch.backends.cuda.enable_cudnn_sdp_python(True)
            self.assertEqual({"flash", "cudnn"}, installed)
            self.assertEqual("FA_FAKE", attention.current_flash_attention_impl())
            self.assertTrue(torch.backends.cuda.cudnn_sdp_python_enabled())

            torch.backends.cuda.enable_cudnn_sdp_python(False)
            self.assertEqual({"flash"}, installed)
            self.assertFalse(torch.backends.cuda.cudnn_sdp_python_enabled())
            self.assertEqual("FA_FAKE", attention.current_flash_attention_impl())

    def test_cudnn_python_enable_is_idempotent(self):
        """Both entry points share one handle, so enabling twice installs once
        and either route can turn it off."""
        calls = []

        def _register():
            calls.append(1)

            class _H:
                def remove(self):
                    calls.clear()

            return _H()

        self.addCleanup(_cudnn.disable)
        with mock.patch.object(_cudnn, "_PROVIDER_REGISTER_FN", _register):
            torch.backends.cuda.enable_cudnn_sdp_python(True)
            handle = _cudnn._ACTIVE_HANDLE
            torch.backends.cuda.enable_cudnn_sdp_python(True)
            self.assertIs(handle, _cudnn._ACTIVE_HANDLE)
            self.assertEqual(1, len(calls))

            # restoring through the registry clears the backends view too
            attention.register_flash_attention_impl(
                "CUDNN", register_fn=_cudnn.register_cudnn_attention
            )
            attention.activate_flash_attention_impl("CUDNN")
            attention.restore_flash_attention_impl()
            self.assertFalse(torch.backends.cuda.cudnn_sdp_python_enabled())

    def test_cudnn_env_var_enables(self):
        """TORCH_CUDNN_SDPA_USE_PYTHON turns the Python implementation on
        without a source change."""
        installed = []

        def _register():
            installed.append(1)

            class _H:
                def remove(self):
                    installed.clear()

            return _H()

        self.addCleanup(_cudnn.disable)
        with mock.patch.object(_cudnn, "_PROVIDER_REGISTER_FN", _register):
            with mock.patch.dict(os.environ, {_cudnn.ENV_VAR: "1"}):
                _cudnn._enable_from_env()
            self.assertTrue(torch.backends.cuda.cudnn_sdp_python_enabled())
            self.assertEqual(1, len(installed))

    def test_cudnn_env_var_unset_is_a_no_op(self):
        for value in ("", "0", "no"):
            with mock.patch.dict(os.environ, {_cudnn.ENV_VAR: value}):
                _cudnn._enable_from_env()
            self.assertFalse(torch.backends.cuda.cudnn_sdp_python_enabled())

    def test_cudnn_env_var_failure_does_not_raise(self):
        """A stale value must not break `import torch`: the variable is
        process-wide and usually set by a job launcher."""

        def _boom():
            raise RuntimeError("provider is broken")

        with mock.patch.object(_cudnn, "_PROVIDER_REGISTER_FN", _boom):
            with mock.patch.dict(os.environ, {_cudnn.ENV_VAR: "1"}):
                _cudnn._enable_from_env()  # must not raise
        self.assertFalse(torch.backends.cuda.cudnn_sdp_python_enabled())


if __name__ == "__main__":
    run_tests()
