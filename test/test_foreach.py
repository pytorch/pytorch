import torch
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
            with self.assertRaisesRegex(RuntimeError, "result type Long can't be cast to the desired output type Bool"):
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

instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()
