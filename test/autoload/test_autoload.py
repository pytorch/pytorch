# Owner(s): ["module: unknown"]

import os
import sys

import torch

from torch.testing._internal.common_utils import run_tests, TestCase


class TestAutoload(TestCase):
    def setUp(self):
        super().setUp()

        self.device_backend_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "device_backend")
        )

        # Add the extension to the system path
        if self.device_backend_path not in sys.path:
            sys.path.insert(0, self.device_backend_path)

    def test_autoload_switch(self):
        # Enabled by default
        self.assertEqual(torch._is_device_backend_autoload_enabled(), True)

    def test_autoload(self):
        # After importing the extension, the value of this environment variable should be true
        torch._import_device_backends()
        value = os.getenv("IS_CUSTOM_DEVICE_BACKEND_IMPORTED", "false")
        self.assertEqual(value, "true")

        # Test the function defined in backend_pkg/__init__.py
        import backend_pkg
        self.assertTrue(hasattr(backend_pkg, "apply_patch"))
        self.assertEqual(backend_pkg.apply_patch(), "success")

    def tearDown(self):
        super().tearDown()

        # Remove the extension from the system path
        if self.device_backend_path in sys.path:
            sys.path.remove(self.device_backend_path)


if __name__ == "__main__":
    run_tests()
