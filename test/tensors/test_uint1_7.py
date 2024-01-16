# Owner(s): ["oncall: quantization"]

import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import run_tests
from torch.tensors import UInt4Tensor

class TestUInt1_7(TestCase):
    def test_uint1_7_dtype(self):

        def up_size(size):
            return (*size[:-1], size[-1] * 2)

        class UInt4TensorTest(torch.Tensor):
            @staticmethod
            def __new__(cls, elem, **kwargs):
                assert elem.dtype is torch.uint8
                assert not kwargs.get("requires_grad", False)
                kwargs["requires_grad"] = False
                return torch.Tensor._make_wrapper_subclass(cls, up_size(elem.shape), dtype=torch.uint4, **kwargs)

            def __init__(self, elem):
                self.elem = elem

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs=None):
                pass

        # make sure it runs
        x = UInt4TensorTest(torch.tensor([
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
        ], dtype=torch.uint8))
        assert x.dtype == torch.uint4

    # TODO: enable other uint types
    def test_constructor(self):
        x = UInt4Tensor(torch.tensor([
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
        ], dtype=torch.uint8))

        self.assertEqual(x.dtype, torch.uint4)
        self.assertEqual(x.shape, (3, 16))
        # making sure these works
        x.to(torch.uint8)
        expected = UInt4Tensor(torch.tensor([
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
        ], dtype=torch.uint8))
        self.assertEqual(x[0:1, :], expected)
        expected = UInt4Tensor(torch.tensor([
            [0x23, 0x45],
            [0x23, 0x45],
            [0x23, 0x45],
        ], dtype=torch.uint8))
        self.assertEqual(x[:, 2:6], expected)

if __name__ == '__main__':
    run_tests()
