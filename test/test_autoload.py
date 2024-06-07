# Owner(s): ["module: unknown"]

import os
import torch

from torch.testing._internal.common_utils import run_tests, TestCase


class TestDeviceBackendAutoload(TestCase):
    def test_autoload(self):
        switch = os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "True")

        # After importing the extension, the value of this environment variable should be true
        # See: test/cpp_extensions/torch_test_cpp_extension/__init__.py
        is_imported = os.getenv("IS_CUSTOM_DEVICE_BACKEND_IMPORTED", "false")
        expected = "true" if bool(switch) else "false"
        self.assertEqual(is_imported, expected)


if __name__ == "__main__":
    run_tests()
