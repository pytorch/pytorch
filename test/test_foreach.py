import torch
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, skipCUDAIfRocm

class TestForeach(TestCase):
    bin_ops = [
        torch._foreach_add,
        torch._foreach_add_,
        torch._foreach_sub,
        torch._foreach_sub_,
        torch._foreach_mul,
        torch._foreach_mul_,
        torch._foreach_div,
        torch._foreach_div_,
    ]

    def _get_test_data(self, device, dtype, N):
        if dtype in [torch.bfloat16, torch.bool, torch.float16]:
            tensors = [torch.randn(N, N, device=device).to(dtype) for _ in range(N)]

        elif dtype in torch.testing.get_all_int_dtypes():
            tensors = [torch.randint(1, 100, (N, N), device=device, dtype=dtype) for _ in range(N)]
        else:
            tensors = [torch.randn(N, N, device=device, dtype=dtype) for _ in range(N)]

        return tensors

    def _test_bin_op_list(self, device, dtype, foreach_op, foreach_op_, torch_op, N=20):
        tensors1 = self._get_test_data(device, dtype, N)
        tensors2 = self._get_test_data(device, dtype, N)

        expected = [torch_op(tensors1[i], tensors2[i]) for i in range(N)]
        res = foreach_op(tensors1, tensors2)
        foreach_op_(tensors1, tensors2)
        self.assertEqual(res, tensors1)
        self.assertEqual(tensors1, expected)

    def _test_unary_op(self, device, dtype, foreach_op, foreach_op_, torch_op, N=20):
        tensors1 = self._get_test_data(device, dtype, N)
        expected = [torch_op(tensors1[i]) for i in range(N)]
        res = foreach_op(tensors1)
        foreach_op_(tensors1)
        self.assertEqual(res, tensors1)
        self.assertEqual(tensors1, expected)

    def _test_pointwise_op(self, device, dtype, foreach_op, foreach_op_, torch_op, N=20):
        tensors = self._get_test_data(device, dtype, N)
        tensors1 = self._get_test_data(device, dtype, N)
        tensors2 = self._get_test_data(device, dtype, N)
        value = 2

        expected = [torch_op(tensors[i], tensors1[i], tensors2[i], value=value) for i in range(N)]

        res = foreach_op(tensors, tensors1, tensors2, value)
        foreach_op_(tensors, tensors1, tensors2, value)
        self.assertEqual(res, tensors)
        self.assertEqual(tensors, expected)

    def _test_bin_op_list_alpha(self, device, dtype, foreach_op, foreach_op_, torch_op, N=20):
        tensors1 = self._get_test_data(device, dtype, N)
        tensors2 = self._get_test_data(device, dtype, N)
        alpha = 2

        expected = [torch_op(tensors1[i], torch.mul(tensors2[i], alpha)) for i in range(N)]
        res = foreach_op(tensors1, tensors2, alpha=alpha)
        foreach_op_(tensors1, tensors2, alpha=alpha)
        self.assertEqual(res, tensors1)

        if dtype == torch.bool:
            expected = [e.to(torch.bool) for e in expected]
        self.assertEqual(tensors1, expected)

    #
    # Unary ops
    #
    @dtypes(*[torch.float, torch.double, torch.complex64, torch.complex128])
    def test_sqrt(self, device, dtype):
        self._test_unary_op(device, dtype, torch._foreach_sqrt, torch._foreach_sqrt_, torch.sqrt)

    @dtypes(*[torch.float, torch.double, torch.complex64, torch.complex128])
    def test_exp(self, device, dtype):
        self._test_unary_op(device, dtype, torch._foreach_exp, torch._foreach_exp_, torch.exp)

    #
    # Pointwise ops
    #
    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=False, include_bool=False, include_complex=False))
    def test_addcmul(self, device, dtype):
        if device == 'cpu':
            if dtype == torch.half:
                with self.assertRaisesRegex(RuntimeError, r"\"addcmul_cpu_out\" not implemented for \'Half\'"):
                    self._test_pointwise_op(device, dtype, torch._foreach_addcmul, 
                                            torch._foreach_addcmul_, torch.addcmul)
                return

        self._test_pointwise_op(device, dtype, torch._foreach_addcmul, torch._foreach_addcmul_, torch.addcmul)

    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=False, include_bool=False, include_complex=False))
    def test_addcdiv(self, device, dtype):
        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            with self.assertRaisesRegex(RuntimeError, 
                                        "Integer division with addcdiv is no longer supported, and in a future"):
                self._test_pointwise_op(device, dtype, torch._foreach_addcdiv, torch._foreach_addcdiv_, torch.addcdiv)
            return

        if device == 'cpu':
            if dtype == torch.half:
                with self.assertRaisesRegex(RuntimeError, r"\"addcdiv_cpu_out\" not implemented for \'Half\'"):
                    self._test_pointwise_op(device, dtype, torch._foreach_addcdiv, 
                                            torch._foreach_addcdiv_, torch.addcdiv)
                return
        self._test_pointwise_op(device, dtype, torch._foreach_addcdiv, torch._foreach_addcdiv_, torch.addcdiv)

    #
    # Ops with scalar
    #
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
    def test_add_with_different_size_tensors(self, device, dtype):
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

    def test_bin_op_scalar_with_different_tensor_dtypes(self, device):
        tensors = [torch.tensor([1.1], dtype=torch.float, device=device), 
                   torch.tensor([1], dtype=torch.long, device=device)]
        self.assertRaises(RuntimeError, lambda: torch._foreach_add(tensors, 1))

    #
    # Ops with list
    #
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
    def test_add_list(self, device, dtype):
        self._test_bin_op_list(device, dtype, torch._foreach_add, torch._foreach_add_, torch.add)
        self._test_bin_op_list_alpha(device, dtype, torch._foreach_add, torch._foreach_add_, torch.add)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_sub_list(self, device, dtype):
        if dtype == torch.bool:
            with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with two bool"):
                self._test_bin_op_list(device, dtype, torch._foreach_sub, torch._foreach_sub_, torch.sub)

            with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool tensor"):
                self._test_bin_op_list_alpha(device, dtype, torch._foreach_sub, torch._foreach_sub_, torch.sub)
        else:
            self._test_bin_op_list(device, dtype, torch._foreach_sub, torch._foreach_sub_, torch.sub)
            self._test_bin_op_list_alpha(device, dtype, torch._foreach_sub, torch._foreach_sub_, torch.sub)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_mul_list(self, device, dtype):
        self._test_bin_op_list(device, dtype, torch._foreach_mul, torch._foreach_mul_, torch.mul)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_div_list(self, device, dtype):
        if dtype in torch.testing.integral_types_and(torch.bool):
            with self.assertRaisesRegex(RuntimeError, "Integer division of tensors using div or / is no longer"):
                self._test_bin_op_list(device, dtype, torch._foreach_div, torch._foreach_div_, torch.div)
            return

        self._test_bin_op_list(device, dtype, torch._foreach_div, torch._foreach_div_, torch.div)

    def test_bin_op_list_error_cases(self, device):
        tensors1 = []
        tensors2 = []

        for bin_op in self.bin_ops: 
            # Empty lists
            with self.assertRaises(RuntimeError):
                bin_op(tensors1, tensors2)

            # One empty list
            tensors1.append(torch.tensor([1], device=device))
            with self.assertRaises(RuntimeError):
                bin_op(tensors1, tensors2)

            # Lists have different amount of tensors
            tensors2.append(torch.tensor([1], device=device))
            tensors2.append(torch.tensor([1], device=device))
            with self.assertRaises(RuntimeError):
                bin_op(tensors1, tensors2)

            # Different dtypes
            tensors1 = [torch.zeros(2, 2, device=device, dtype=torch.float) for _ in range(2)]
            tensors2 = [torch.ones(2, 2, device=device, dtype=torch.int) for _ in range(2)]

            with self.assertRaises(RuntimeError):
                bin_op(tensors1, tensors2)

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
