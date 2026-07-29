# Owner(s): ["module: dsl-native-ops"]

import torch
import torch.backends.python_native as pn
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFlyDSLRegistry(TestCase):
    def test_flydsl_backend_is_exposed(self):
        self.assertIsNotNone(torch)
        self.assertIn("flydsl", pn.all_dsls)
        self.assertTrue(hasattr(pn, "flydsl"))


if __name__ == "__main__":
    run_tests()
