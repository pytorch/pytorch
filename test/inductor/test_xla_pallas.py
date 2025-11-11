import unittest
import torch
from torch.testing._internal.common_utils import TestCase
from torch._inductor.test_case import run_tests
from torch.utils._pallas import has_pallas

HAS_PALLAS = has_pallas()

class PallasXlaTests(TestCase):
    @unittest.skipUnless(HAS_PALLAS, "Pallas is not available")
    def test_xla_add(self):
        def fn(a, b):
            return a + b

        a = torch.randn(10, device="xla")
        b = torch.randn(10, device="xla")

        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a, b)
        expected = a + b

        self.assertEqual(result, expected)

if __name__ == "__main__":
    run_tests()
