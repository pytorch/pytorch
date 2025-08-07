import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class TestOpenRegAutoload(TestCase):
    """Tests of autoloading the OpenReg backend"""

    def test_autoload_disable(self):
        # Test that the backend is loaded automatically
        self.assertFalse(hasattr(torch, "openreg"))
        import torch_openreg # noqa: F401
        self.assertTrue(torch.openreg.is_available())


if __name__ == "__main__":
    run_tests()