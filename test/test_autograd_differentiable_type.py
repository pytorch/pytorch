import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestIsDifferentiableType(TestCase):
    def test_is_differentiable_type(self):
        differentiable = (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.complex64,
            torch.complex128,
        )
        non_differentiable = (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        )

        for dtype in differentiable:
            self.assertTrue(torch._C._is_differentiable_type(dtype))
        for dtype in non_differentiable:
            self.assertFalse(torch._C._is_differentiable_type(dtype))

    def test_is_differentiable_type_requires_dtype(self):
        with self.assertRaisesRegex(TypeError, "dtype must be a torch.dtype"):
            torch._C._is_differentiable_type("float32")


if __name__ == "__main__":
    run_tests()
