# Owner(s): ["module: meta tensors"]

from torch.testing._internal.common_utils import (
    TestCase, run_tests
)
import torch
from torch._subclasses.dynamic_tensor import DynamicTensor

class DynamicTensorTest(TestCase):
    def test_basic(self):
        r = DynamicTensor.wrap(torch.ones(2, 2), dim=0)
        self.assertExpectedInline(r, '''\
DynamicTensor.wrap(tensor([[1., 1.],
        [1., 1.]]), dim=(0,))''')
        self.assertEqual(r.dynamic_dims, (0,))
        r2 = r + r
        self.assertEqual(r2.dynamic_dims, (0,))

        r = DynamicTensor.wrap(torch.randn(1, 2), dim=0)
        self.assertEqual(r.dynamic_dims, (0,))
        r + torch.randn(1, 2)  # OK, because RHS broadcasts
        torch.randn(1, 2) + r  # OK, because LHS broadcasts
        self.assertRaises(RuntimeError, lambda: r + torch.randn(5, 2))  # cannot broadcast

        r2 = DynamicTensor.wrap(torch.randn(1, 2), dim=0)
        self.assertRaises(RuntimeError, lambda: r + r2)  # unrelated

if __name__ == "__main__":
    run_tests()
