import unittest
import torch
from torch._dynamo.testing import same

class TestSameAllclose(unittest.TestCase):
    def test_same_allclose_exception(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        res = torch.tensor([1.0, 2.0, float("nan")])

        try:
            same(ref, res, equal_nan=False)
        except RuntimeError as e:
            self.assertIn(
                "An unexpected error occurred while comparing tensors with torch.allclose",
                str(e),
            )
            self.assertIn(
                "RuntimeError: The size of tensor a (3) must match the size of tensor b (3) "
                "at non-singleton dimension 0",
                str(e),
            )
        else:
            self.fail("RuntimeError not raised")

    def test_same_allclose_no_exception(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        res = torch.tensor([1.0, 2.0, 3.0])

        self.assertTrue(same(ref, res))

if __name__ == "__main__":
    unittest.main()
