# Owner(s): ["module: sdpa"]

import torch.nn.attention as attention
from torch.nn.attention import _registry
from torch.testing._internal.common_utils import run_tests, TestCase


class FakeHandle:
    def remove(self):
        pass


class TestFlashAttentionRegistry(TestCase):
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


if __name__ == "__main__":
    run_tests()
