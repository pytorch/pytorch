# Owner(s): ["module: dsl-native-ops"]

import torch
from torch._native.const_tensor_wrapper import ConstTensorWrapper
from torch._native.utils.tensor import const_data_ptr, reshape_contiguous
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestTensorPreparation(TestCase):
    @dtypes(torch.float32, torch.complex64)
    @parametrize(
        "layout",
        ["contiguous", "reshape", "strided", "offset", "negative", "conjugate"],
    )
    def test_reshape_contiguous(
        self, device: str, dtype: torch.dtype, layout: str
    ) -> None:
        x = torch.randn(4, 8, device=device, dtype=dtype)
        shape = (2, 16) if layout == "reshape" else (4, 8)
        if layout == "strided":
            x = torch.randn(4, 16, device=device, dtype=dtype)[:, ::2]
        elif layout == "offset":
            x = torch.randn(33, device=device, dtype=dtype)[1:].view(4, 8)
        elif layout == "negative":
            x = x._neg_view()
        elif layout == "conjugate":
            x = x.conj()
        actual = reshape_contiguous(x, shape, alignment=16)
        self.assertEqual(actual, x.reshape(shape))
        self.assertTrue(actual.is_contiguous())
        self.assertFalse(actual.is_neg() or actual.is_conj())
        self.assertEqual(const_data_ptr(actual) % 16, 0)
        if layout == "contiguous":
            self.assertIs(actual, x)
        elif layout == "reshape":
            self.assertEqual(const_data_ptr(actual), const_data_ptr(x))

    @dtypes(torch.float32)
    def test_read_only_preserves_cow(self, device: str, dtype: torch.dtype) -> None:
        x = torch.randn(4, 8, device=device, dtype=dtype)._lazy_clone()
        pointer = const_data_ptr(x)
        self.assertEqual(ConstTensorWrapper(x).data_ptr(), pointer)
        y = reshape_contiguous(x, (2, 16), alignment=16)
        self.assertEqual(const_data_ptr(y), pointer)
        self.assertTrue(torch._C._is_cow_tensor(x))
        self.assertTrue(torch._C._is_cow_tensor(y))


instantiate_device_type_tests(
    TestTensorPreparation, globals(), only_for=("cpu", "cuda")
)


if __name__ == "__main__":
    run_tests()
