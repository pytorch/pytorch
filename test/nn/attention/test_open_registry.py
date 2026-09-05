# Owner(s): ["module: sdpa"]

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
            _cudnn.register_flash_attention_cudnn("cudnn_frontend_not_installed_xyz")
        self.assertIsNone(attention.current_flash_attention_impl())

    def test_cudnn_package_without_registry_support_raises(self):
        """A provider too old to register itself must produce a clear error
        instead of recursing back into this shim: the registered callable is
        still the shim after the import, so calling it again would loop."""
        attention.register_flash_attention_impl(
            "CUDNN", register_fn=_cudnn.register_flash_attention_cudnn
        )
        # types is always importable and registers nothing.
        with self.assertRaisesRegex(RuntimeError, "did not register"):
            _cudnn.register_flash_attention_cudnn("types")


if __name__ == "__main__":
    run_tests()
