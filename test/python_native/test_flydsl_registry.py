# Owner(s): ["module: dsl-native-ops"]

import subprocess
import sys

import torch
import torch.backends.python_native as pn
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFlyDSLRegistry(TestCase):
    def test_flydsl_is_registered(self):
        self.assertIsNotNone(torch)
        self.assertIn("flydsl", pn.all_dsls)
        self.assertTrue(hasattr(pn, "flydsl"))

    def test_import_torch_does_not_import_flydsl_runtime(self):
        script = """
import sys
import torch
print("flydsl" in sys.modules)
"""
        result = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
        self.assertEqual(result.rsplit("\\n", 1)[-1], "False")


if __name__ == "__main__":
    run_tests()
