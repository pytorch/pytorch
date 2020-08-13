import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes

class TestForeach(TestCase):
    # Unary ops
    @dtypes(*[torch.int32, torch.half, torch.float, torch.double, torch.complex64, torch.complex128])
    def test_sqrt(self, device, dtype):
        if dtype in [torch.bool, torch.int,torch.half,]:
            return
        tensors = [torch.ones(20, 20, device=device, dtype=dtype) for _ in range(20)]

        res = torch._foreach_sqrt(tensors)
        torch._foreach_sqrt_(tensors)

        self.assertEqual([torch.sqrt(torch.ones(20, 20, device=device, dtype=dtype)) for _ in range(20)], res)
        self.assertEqual(tensors, res)

    @dtypes(*[torch.int32, torch.half, torch.float, torch.double, torch.complex64, torch.complex128])
    def test_exp(self, device, dtype):
        if dtype in [torch.bool, torch.int,torch.half,]:
            return

        tensors = [torch.ones(20, 20, device=device, dtype=dtype) for _ in range(20)]

        res = torch._foreach_exp(tensors)
        torch._foreach_exp_(tensors)

        self.assertEqual([torch.exp(torch.ones(20, 20, device=device, dtype=dtype)) for _ in range(20)], res)
        self.assertEqual(tensors, res)

    # Ops with scalar
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
        complex_scalar = 3+5j

        if dtype == torch.bfloat16:
            # bug: 42374
            self.assertRaises(RuntimeError, lambda: torch._foreach_add(tensors, complex_scalar))
            return

        # bool tensor + 1 will result in int64 tensor
        expected = [torch.add(complex_scalar, torch.zeros(10, 10, device=device, dtype=dtype)) for _ in range(10)]

        if dtype in [torch.float16, torch.float32, torch.float64] and device == 'cuda:0':
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
        tensors = [torch.zeros(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]
        self.assertRaises(RuntimeError, lambda: torch._foreach_add(tensors, 1))

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

    @dtypes(*torch.testing.get_all_dtypes())
    def test_sub_scalar_same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            # Subtraction, the `-` operator, with a bool tensor is not supported.
            return

        tensors = [torch.ones(20, 20, device=device, dtype=dtype) for _ in range(20)]
        res = torch._foreach_sub(tensors, 1)
        for t in res:
            if dtype == torch.bool and device == 'cpu':
                dtype = torch.int64
            self.assertEqual(t, torch.zeros(20, 20, device=device, dtype=dtype))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_sub_scalar__same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            # Subtraction, the `-` operator, with a bool tensor is not supported.
            return

        tensors = [torch.ones(20, 20, device=device, dtype=dtype) for _ in range(20)]
        torch._foreach_sub_(tensors, 1)
        for t in tensors:
            if dtype == torch.bool and device == 'cpu':
                dtype = torch.int64
            self.assertEqual(t, torch.zeros(20, 20, device=device, dtype=dtype))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_mul_scalar_same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return

        tensors = [torch.ones(20, 20, device=device, dtype=dtype) for _ in range(20)]
        res = torch._foreach_mul(tensors, 3)
        for t in res:
            self.assertEqual(t, torch.ones(20, 20, device=device, dtype=dtype).mul(3))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_mul_scalar__same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return

        tensors = [torch.ones(20, 20, device=device, dtype=dtype) for _ in range(20)]
        torch._foreach_mul_(tensors, 3)
        for t in tensors:
            self.assertEqual(t, torch.ones(20, 20, device=device, dtype=dtype).mul(3))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_div_scalar_same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return

        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            # Integer division of tensors using div or / is no longer supported
            return

        tensors = [torch.ones(20, 20, device=device, dtype=dtype) for _ in range(20)]
        res = torch._foreach_div(tensors, 2)
        for t in res:
            self.assertEqual(t, torch.ones(20, 20, device=device, dtype=dtype).div(2))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_div_scalar__same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return

        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            # Integer division of tensors using div or / is no longer supported
            return

        tensors = [torch.ones(20, 20, device=device, dtype=dtype) for _ in range(20)]
        torch._foreach_div_(tensors, 2)
        for t in tensors:
            self.assertEqual(t, torch.ones(20, 20, device=device, dtype=dtype).div(2))

    # Ops with list
    @dtypes(*torch.testing.get_all_dtypes())
    def test_bin_op_list_same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return

        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            # Integer division of tensors using div or / is no longer supported
            return

        tensors1 = []
        tensors2 = []
        for _ in range(20):
            tensors1.append(torch.zeros(20, 20, device=device, dtype=dtype))
            tensors2.append(torch.ones(20, 20, device=device, dtype=dtype))

        res = torch._foreach_mul(tensors1, tensors2)
        for t in res:
            self.assertEqual(t, torch.zeros(20, 20, device=device, dtype=dtype))

        res = torch._foreach_div(torch._foreach_add(tensors1, 4), torch._foreach_mul(tensors2, 2))
        for t in res:
            self.assertEqual(t, torch.ones(20, 20, device=device, dtype=dtype).mul(2))

        res = torch._foreach_add(tensors1, tensors2)
        for t in res:
            self.assertEqual(t, torch.ones(20, 20, device=device, dtype=dtype))

        res = torch._foreach_sub(res, tensors2)
        for t in res:
            self.assertEqual(t, torch.zeros(20, 20, device=device, dtype=dtype))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_bin_op_list__same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return

        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            # Integer division of tensors using div or / is no longer supported
            return

        tensors1 = []
        tensors2 = []
        for _ in range(20):
            tensors1.append(torch.zeros(20, 20, device=device, dtype=dtype))
            tensors2.append(torch.ones(20, 20, device=device, dtype=dtype))


        torch._foreach_add_(tensors1, tensors2)
        for t in tensors1:
            self.assertEqual(t, torch.ones(20, 20, device=device, dtype=dtype))

        torch._foreach_sub_(tensors1, tensors2)
        for t in tensors1:
            self.assertEqual(t, torch.zeros(20, 20, device=device, dtype=dtype))

        torch._foreach_mul_(tensors1, tensors2)
        for t in tensors1:
            self.assertEqual(t, torch.zeros(20, 20, device=device, dtype=dtype))

        torch._foreach_add_(tensors1, 4)
        torch._foreach_add_(tensors2, 1)
        torch._foreach_div_(tensors1, tensors2)
        for t in tensors1:
            self.assertEqual(t, torch.ones(20, 20, device=device, dtype=dtype).mul(2))

    def test_add_list_error_cases(self, device):
        tensors1 = []
        tensors2 = []

        # Empty lists
        with self.assertRaises(RuntimeError):
            torch._foreach_add(tensors1, tensors2)
            torch._foreach_add_(tensors1, tensors2)

        # One empty list
        tensors1.append(torch.tensor([1], device=device))
        with self.assertRaises(RuntimeError):
            torch._foreach_add(tensors1, tensors2)
            torch._foreach_add_(tensors1, tensors2)

        # Lists have different amount of tensors
        tensors2.append(torch.tensor([1], device=device))
        tensors2.append(torch.tensor([1], device=device))
        with self.assertRaises(RuntimeError):
            torch._foreach_add(tensors1, tensors2)
            torch._foreach_add_(tensors1, tensors2)

        # Different dtypes
        tensors1 = []
        tensors2 = []
        for _ in range(10):
            tensors1.append(torch.zeros(10, 10, device=device, dtype=torch.float))
            tensors2.append(torch.ones(10, 10, device=device, dtype=torch.int))

        with self.assertRaises(RuntimeError):
            torch._foreach_add(tensors1, tensors2)
            torch._foreach_add_(tensors1, tensors2)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_list_same_size(self, device, dtype):
        tensors1 = []
        tensors2 = []
        for _ in range(10):
            tensors1.append(torch.zeros(10, 10, device=device, dtype=dtype))
            tensors2.append(torch.ones(10, 10, device=device, dtype=dtype))

        res = torch._foreach_add(tensors1, tensors2)
        torch._foreach_add_(tensors1, tensors2)
        self.assertEqual(res, tensors1)
        self.assertEqual(res[0], torch.ones(10, 10, device=device, dtype=dtype))

instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()
