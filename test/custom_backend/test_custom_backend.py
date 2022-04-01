# Owner(s): ["module: unknown"]

import os
import tempfile
import torch

from backend import Model, to_custom_backend, get_custom_backend_library_path
from torch.testing._internal.common_utils import TestCase, run_tests


class TestCustomBackend(TestCase):
    def setUp(self):
        # Load the library containing the custom backend.
        self.library_path = get_custom_backend_library_path()
        torch.ops.load_library(self.library_path)
        # Create an instance of the test Module and lower it for
        # the custom backend.
        self.model = to_custom_backend(torch.jit.script(Model()))

    def test_execute(self):
        """
        Test execution using the custom backend.
        """
        a = torch.randn(4)
        b = torch.randn(4)
        # The custom backend is hardcoded to compute f(a, b) = (a + b, a - b).
        expected = (a + b, a - b)
        out = self.model(a, b)
        self.assertTrue(expected[0].allclose(out[0]))
        self.assertTrue(expected[1].allclose(out[1]))

    def test_save_load(self):
        """
        Test that a lowered module can be executed correctly
        after saving and loading.
        """
        # Test execution before saving and loading to make sure
        # the lowered module works in the first place.
        self.test_execute()

        # Save and load.
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.close()
            torch.jit.save(self.model, f.name)
            loaded = torch.jit.load(f.name)
        finally:
            os.unlink(f.name)
        self.model = loaded

        # Test execution again.
        self.test_execute()


if __name__ == "__main__":
    run_tests()
