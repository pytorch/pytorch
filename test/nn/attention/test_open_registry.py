# Owner(s): ["module: sdpa"]

import torch.nn.attention as attention
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFlashAttentionRegistry(TestCase):
    def setUp(self):
        super().setUp()
        self._saved_backends = dict(attention._FLASH_ATTENTION_BACKENDS)
        self._saved_active = attention.current_flash_attention_backend()
        attention._FLASH_ATTENTION_BACKENDS.clear()
        attention._FLASH_ATTENTION_ACTIVE = None

    def tearDown(self):
        attention._FLASH_ATTENTION_BACKENDS.clear()
        attention._FLASH_ATTENTION_BACKENDS.update(self._saved_backends)
        attention._FLASH_ATTENTION_ACTIVE = self._saved_active
        super().tearDown()

    def test_register_and_install_backend(self):
        calls: dict[str, bool] = {}

        def fake_register():
            calls["called"] = True

        attention.register_flash_attention_backend("TEST_FA", register_fn=fake_register)
        self.assertIn("TEST_FA", attention.list_flash_attention_backends())

        attention.install_flash_attention_impl("TEST_FA")

        self.assertTrue(calls.get("called", False))
        self.assertEqual("TEST_FA", attention.current_flash_attention_backend())

    def test_install_unknown_backend_errors(self):
        with self.assertRaisesRegex(
            ValueError, "Unknown flash attention backend 'missing'"
        ):
            attention.install_flash_attention_impl("missing")


if __name__ == "__main__":
    run_tests()
