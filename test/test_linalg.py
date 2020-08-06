import torch
import unittest

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_NUMPY)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes)

if TEST_NUMPY:
    import numpy as np

class TestLinalg(TestCase):
    exact_dtype = True

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @dtypes(torch.float)
    def test_outer(self, device, dtype):
        a = torch.randn(50, device=device, dtype=dtype)
        b = torch.randn(50, device=device, dtype=dtype)

        def fn(a, b):
            return torch.linalg.outer(a, b)
        scripted_outer = torch.jit.script(fn)

        expected = np.outer(a.cpu().numpy(), b.cpu().numpy())
        original = torch.ger(a, b)
        actual = torch.linalg.outer(a, b)
        scripted = scripted_outer(a, b)

        self.assertEqual(actual, expected, exact_device=False)
        self.assertEqual(actual, original)
        self.assertEqual(actual, scripted)


instantiate_device_type_tests(TestLinalg, globals())

if __name__ == '__main__':
    run_tests()
