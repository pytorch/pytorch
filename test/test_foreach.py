import torch
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes

class TestForeach(TestCase):
    @dtypes(*torch.testing.get_all_dtypes())
    def test_int_scalar(self, device, dtype):
        tensors = [torch.zeros(10, 10, device=device, dtype=dtype) for _ in range(10)]
        int_scalar = 1

        # bool tensor + 1 will result in int64 tensor
        if dtype == torch.bool:
            expected = [torch.ones(10, 10, device=device, dtype=torch.int64) for _ in range(10)]
        else:
            expected = [torch.ones(10, 10, device=device, dtype=dtype) for _ in range(10)]

        res = torch._foreach_add(tensors, int_scalar)
        self.assertEqual(res, expected)

        if dtype in [torch.bool]:
            with self.assertRaisesRegex(RuntimeError, 
                                        "result type Long can't be cast to the desired output type Bool"):
                torch._foreach_add_(tensors, int_scalar)
        else:
            torch._foreach_add_(tensors, int_scalar)
            self.assertEqual(res, tensors)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_float_scalar(self, device, dtype):
        tensors = [torch.zeros(10, 10, device=device, dtype=dtype) for _ in range(10)]
        float_scalar = 1.

        # float scalar + integral tensor will result in float tensor
        if dtype in [torch.uint8, torch.int8, torch.int16, 
                     torch.int32, torch.int64, torch.bool]:
            expected = [torch.ones(10, 10, device=device, dtype=torch.float32) for _ in range(10)]
        else:
            expected = [torch.ones(10, 10, device=device, dtype=dtype) for _ in range(10)]

        res = torch._foreach_add(tensors, float_scalar)
        self.assertEqual(res, expected)

        if dtype in [torch.uint8, torch.int8, torch.int16, 
                     torch.int32, torch.int64, torch.bool]:
            self.assertRaises(RuntimeError, lambda: torch._foreach_add_(tensors, float_scalar))
        else:
            torch._foreach_add_(tensors, float_scalar)
            self.assertEqual(res, tensors)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_complex_scalar(self, device, dtype):
        tensors = [torch.zeros(10, 10, device=device, dtype=dtype) for _ in range(10)]
        complex_scalar = 3 + 5j

        # bool tensor + 1 will result in int64 tensor
        expected = [torch.add(complex_scalar, torch.zeros(10, 10, device=device, dtype=dtype)) for _ in range(10)]

        if dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16] and device == 'cuda:0':
            # value cannot be converted to dtype without overflow: 
            self.assertRaises(RuntimeError, lambda: torch._foreach_add_(tensors, complex_scalar))
            self.assertRaises(RuntimeError, lambda: torch._foreach_add(tensors, complex_scalar))
            return

        res = torch._foreach_add(tensors, complex_scalar)
        self.assertEqual(res, expected)

        if dtype not in [torch.complex64, torch.complex128]:
            self.assertRaises(RuntimeError, lambda: torch._foreach_add_(tensors, complex_scalar))
        else:
            torch._foreach_add_(tensors, complex_scalar)
            self.assertEqual(res, tensors)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_bool_scalar(self, device, dtype):
        tensors = [torch.zeros(10, 10, device=device, dtype=dtype) for _ in range(10)]
        bool_scalar = True

        expected = [torch.ones(10, 10, device=device, dtype=dtype) for _ in range(10)]

        res = torch._foreach_add(tensors, bool_scalar)
        self.assertEqual(res, expected)

        torch._foreach_add_(tensors, bool_scalar)
        self.assertEqual(res, tensors)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_different_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return

        tensors = [torch.zeros(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]
        expected = [torch.ones(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]
        torch._foreach_add_(tensors, 1)
        self.assertEqual(expected, tensors)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_empty_list_and_empty_tensor(self, device, dtype):
        # TODO: enable empty list case
        for tensors in [[torch.randn([0])]]:
            res = torch._foreach_add(tensors, 1)
            self.assertEqual(res, tensors)

            torch._foreach_add_(tensors, 1)
            self.assertEqual(res, tensors)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_overlapping_tensors(self, device, dtype):
        tensors = [torch.ones(1, 1, device=device, dtype=dtype).expand(2, 1, 3)]
        expected = [torch.tensor([[[2, 2, 2]], [[2, 2, 2]]], dtype=dtype, device=device)]

        # bool tensor + 1 will result in int64 tensor
        if dtype == torch.bool: 
            expected[0] = expected[0].to(torch.int64).add(1)

        res = torch._foreach_add(tensors, 1)
        self.assertEqual(res, expected)

    def test_add_scalar_with_different_tensor_dtypes(self, device):
        tensors = [torch.tensor([1.1], dtype=torch.float, device=device), 
                   torch.tensor([1], dtype=torch.long, device=device)]
        self.assertRaises(RuntimeError, lambda: torch._foreach_add(tensors, 1))

    def test_add_list_error_cases(self, device):
        tensors1 = []
        tensors2 = []

        # Empty lists
        with self.assertRaises(RuntimeError):
            torch._foreach_add(tensors1, tensors2)
        with self.assertRaises(RuntimeError):
            torch._foreach_add_(tensors1, tensors2)

        # One empty list
        tensors1.append(torch.tensor([1], device=device))
        with self.assertRaisesRegex(RuntimeError, "Tensor list must have at least one tensor."):
            torch._foreach_add(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, "Tensor list must have at least one tensor."):
            torch._foreach_add_(tensors1, tensors2)

        # Lists have different amount of tensors
        tensors2.append(torch.tensor([1], device=device))
        tensors2.append(torch.tensor([1], device=device))
        with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 1 and 2"):
            torch._foreach_add(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 1 and 2"):
            torch._foreach_add_(tensors1, tensors2)

        # Different dtypes
        tensors1 = [torch.zeros(10, 10, device=device, dtype=torch.float) for _ in range(10)]
        tensors2 = [torch.ones(10, 10, device=device, dtype=torch.int) for _ in range(10)]

        with self.assertRaisesRegex(RuntimeError, "All tensors in the tensor list must have the same dtype."):
            torch._foreach_add(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, "All tensors in the tensor list must have the same dtype."):
            torch._foreach_add_(tensors1, tensors2)

        # different devices
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            tensor1 = torch.zeros(10, 10, device="cuda:0")
            tensor2 = torch.ones(10, 10, device="cuda:1")
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                torch._foreach_add([tensor1], [tensor2])
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                torch._foreach_add_([tensor1], [tensor2])

        # Coresponding tensors with different sizes 
        tensors1 = [torch.zeros(10, 10, device=device) for _ in range(10)]
        tensors2 = [torch.ones(11, 11, device=device) for _ in range(10)]
        with self.assertRaisesRegex(RuntimeError, "Corresponding tensors in lists must have the same size"):
            torch._foreach_add(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, r", got \[10, 10\] and \[11, 11\]"):
            torch._foreach_add_(tensors1, tensors2)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_list_same_size(self, device, dtype):
        tensors1 = [torch.zeros(10, 10, device=device, dtype=dtype) for _ in range(10)]
        tensors2 = [torch.ones(10, 10, device=device, dtype=dtype) for _ in range(10)]

        res = torch._foreach_add(tensors1, tensors2)
        torch._foreach_add_(tensors1, tensors2)
        self.assertEqual(res, tensors1)
        self.assertEqual(res[0], torch.ones(10, 10, device=device, dtype=dtype))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_list_different_sizes(self, device, dtype):
        tensors1 = [torch.zeros(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]
        tensors2 = [torch.ones(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]

        res = torch._foreach_add(tensors1, tensors2)
        torch._foreach_add_(tensors1, tensors2)
        self.assertEqual(res, tensors1)
        self.assertEqual(res, [torch.ones(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not found")
    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_list_slow_path(self, device, dtype):
        # different strides
        tensor1 = torch.zeros(10, 10, device=device, dtype=dtype)
        tensor2 = torch.ones(10, 10, device=device, dtype=dtype)
        res = torch._foreach_add([tensor1], [tensor2.t()])
        torch._foreach_add_([tensor1], [tensor2])
        self.assertEqual(res, [tensor1])

        # non contiguous 
        tensor1 = torch.randn(5, 2, 1, 3, device=device)[:, 0]
        tensor2 = torch.randn(5, 2, 1, 3, device=device)[:, 0]
        self.assertFalse(tensor1.is_contiguous())
        self.assertFalse(tensor2.is_contiguous())
        res = torch._foreach_add([tensor1], [tensor2])
        torch._foreach_add_([tensor1], [tensor2])
        self.assertEqual(res, [tensor1])

instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()
