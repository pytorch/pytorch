# Owner(s): ["module: PrivateUse1"]

import os

from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class TestDeviceBackendAutoload(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_autoload(self):
        switch = os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

        # After importing the extension, the value of this environment variable should be true
        # See: test/cpp_extensions/torch_test_cpp_extension/__init__.py
        is_imported = os.getenv("IS_CUSTOM_DEVICE_BACKEND_IMPORTED", "0")

        # Both values should be equal
        self.assertEqual(is_imported, switch)


if __name__ == "__main__":
    run_tests()
