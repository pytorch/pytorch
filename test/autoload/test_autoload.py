# Owner(s): ["module: unknown"]

import os
import sys

import torch

from torch.testing._internal.common_utils import run_tests, TestCase


class TestAutoload(TestCase):
    def test_load_plugins(self):
        device_backend_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "device_backend")
        )

        # add the extension to the system path
        if device_backend_path not in sys.path:
            sys.path.insert(0, device_backend_path)

        # after importing the extension, the value of this environment variable should be true
        torch.import_device_backends()
        value = os.getenv("IS_CUSTOM_DEVICE_BACKEND_IMPORTED")
        self.assertEqual(value, "true")

        # remove the extension from the system path
        if device_backend_path in sys.path:
            sys.path.remove(device_backend_path)


if __name__ == "__main__":
    run_tests()
