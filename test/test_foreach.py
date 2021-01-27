import torch
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_ROCM, TEST_WITH_SLOW
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, skipCUDAIfRocm, ops)
from torch._six import inf, nan
from torch.testing._internal.common_methods_invocations import foreach_unary_op_db, foreach_binary_op_db

N_values = [20] if not TEST_WITH_SLOW else [30, 300]

class TestForeach(TestCase):
    bin_ops = [
        (torch._foreach_add, torch._foreach_add_, torch.add),
        (torch._foreach_sub, torch._foreach_sub_, torch.sub),
        (torch._foreach_mul, torch._foreach_mul_, torch.mul),
        (torch._foreach_div, torch._foreach_div_, torch.div),
    ]

    def _get_test_data(self, device, dtype, N):
        if dtype in [torch.bfloat16, torch.bool, torch.float16]:
            tensors = [torch.randn(N, N, device=device).to(dtype) for _ in range(N)]
        elif dtype in torch.testing.get_all_int_dtypes():
            tensors = [torch.randint(1, 100, (N, N), device=device, dtype=dtype) for _ in range(N)]
        else:
            tensors = [torch.randn(N, N, device=device, dtype=dtype) for _ in range(N)]

        return tensors

    def _test_bin_op_list(self, device, dtype, foreach_op, foreach_op_, torch_op):
        for N in N_values:
            tensors1 = self._get_test_data(device, dtype, N)
            tensors2 = self._get_test_data(device, dtype, N)

            # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
            control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                              (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype
            expected = [torch_op(tensors1[i].to(dtype=control_dtype),
                                 tensors2[i].to(dtype=control_dtype)).to(dtype=dtype) for i in range(N)]
            res = foreach_op(tensors1, tensors2)
            foreach_op_(tensors1, tensors2)
            self.assertEqual(res, tensors1)
            if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                self.assertEqual(tensors1, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
            else:
                self.assertEqual(tensors1, expected)

    def _test_pointwise_op(self, device, dtype, foreach_op, foreach_op_, torch_op):
        for N in N_values:
            values = [2 + i for i in range(N)]
            for vals in [values[0], values]:
                tensors = self._get_test_data(device, dtype, N)
                tensors1 = self._get_test_data(device, dtype, N)
                tensors2 = self._get_test_data(device, dtype, N)

                # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
                control_dtype = torch.float32 if (self.device_type == 'cuda' and 
                                                  (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype

                if not isinstance(vals, list):
                    expected = [torch_op(tensors[i].to(dtype=control_dtype),
                                         tensors1[i].to(dtype=control_dtype),
                                         tensors2[i].to(dtype=control_dtype),
                                         value=values[0]).to(dtype=dtype) for i in range(N)]
                else:
                    expected = [torch_op(tensors[i].to(dtype=control_dtype),
                                         tensors1[i].to(dtype=control_dtype),
                                         tensors2[i].to(dtype=control_dtype),
                                         value=values[i]).to(dtype=dtype) for i in range(N)]

                res = foreach_op(tensors, tensors1, tensors2, vals)
                foreach_op_(tensors, tensors1, tensors2, vals)
                self.assertEqual(res, tensors)

                if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(tensors, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                else:
                    self.assertEqual(tensors, expected)

                # test error cases
                for op in [torch._foreach_addcmul, torch._foreach_addcmul_, torch._foreach_addcdiv, torch._foreach_addcdiv_]:
                    tensors = self._get_test_data(device, dtype, N)
                    tensors1 = self._get_test_data(device, dtype, N)
                    tensors2 = self._get_test_data(device, dtype, N)

                    with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N + 1)])

                    with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N - 1)])

                    tensors = self._get_test_data(device, dtype, N + 1)
                    with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 21 and 20"):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N)])

                    tensors1 = self._get_test_data(device, dtype, N + 1)
                    with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 21 and 20"):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N)])

    def _test_bin_op_list_alpha(self, device, dtype, foreach_op, foreach_op_, torch_op):
        for N in N_values:
            tensors1 = self._get_test_data(device, dtype, N)
            tensors2 = self._get_test_data(device, dtype, N)
            alpha = 2

            # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
            control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                              (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype
            expected = [torch_op(tensors1[i].to(dtype=control_dtype),
                                 torch.mul(tensors2[i].to(dtype=control_dtype),
                                 alpha)).to(dtype=dtype) for i in range(N)]
            res = foreach_op(tensors1, tensors2, alpha=alpha)
            foreach_op_(tensors1, tensors2, alpha=alpha)
            self.assertEqual(res, tensors1)

            if dtype == torch.bool:
                expected = [e.to(torch.bool) for e in expected]
            if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                self.assertEqual(tensors1, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
            else:
                self.assertEqual(tensors1, expected)

    @ops(foreach_unary_op_db)
    def test_unary_ops(self, device, dtype, op):
        for N in N_values:
            tensors = op.sample_inputs(device, dtype, N)
            ref_res = [op.ref(t) for t in tensors]

            method = op.get_method()
            inplace = op.get_inplace()
            fe_res = method(tensors)
            self.assertEqual(ref_res, fe_res)

            if op.safe_casts_outputs and dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                    inplace(tensors)
            elif dtype in [torch.complex64, torch.complex128] and inplace == torch._foreach_abs_:
                # Special case for abs
                with self.assertRaisesRegex(RuntimeError, r"In-place abs is not supported for complex tensors."):
                    inplace(tensors)
            else:
                inplace(tensors)
                self.assertEqual(tensors, fe_res)

    # Test foreach binary ops with a single scalar
    # Compare results agains reference torch functions
    # In case of an exeption, check if torch reference function throws as well.
    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_ops_scalar(self, device, dtype, op):
        scalars = [2, 2.2, True, 3 + 5j]

        # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
        dtype = torch.float32 if (self.device_type == 'cuda' and 
                                  (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype
        for N in N_values:
            for scalar in scalars: 
                # test out of place
                foreach_exeption = False
                torch_exeption = False
                tensors = op.sample_inputs(device, dtype, N)
                method = op.get_method()
                inplace = op.get_inplace()

                try:
                    ref_res = [op.ref(t, scalar) for t in tensors]
                except Exception:
                    torch_exeption = True

                try:
                    fe_res = method(tensors, scalar)
                except Exception:
                    foreach_exeption = True

                self.assertEqual(foreach_exeption, torch_exeption)

                if not torch_exeption:
                    if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                        self.assertEqual(ref_res, fe_res, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                    else:
                        self.assertEqual(ref_res, fe_res)

                # test inplace
                foreach_inplace_exeption = False
                torch_inplace_exeption = False
                try:
                    inplace(tensors, scalar)
                    self.assertEqual(tensors, fe_res)
                except Exception:
                    foreach_inplace_exeption = True

                try:
                    # get torch inplace reference function
                    inplace_name = op.ref_name + "_"
                    torch_inplace = getattr(torch.Tensor, inplace_name, None)

                    for t in tensors:
                        torch_inplace(t, scalar)
                except Exception:
                    torch_inplace_exeption = True

                self.assertEqual(foreach_inplace_exeption, torch_inplace_exeption)

    # Test foreach binary ops with a scalar list
    # Compare results agains reference torch functions
    # In case of an exeption, check if torch reference function throws as well.
    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_ops_scalar_list(self, device, dtype, op):
        # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
        dtype = torch.float32 if (self.device_type == 'cuda' and 
                                  (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype
        for N in N_values:
            scalar_lists = [
                [2 for _ in range(N)],
                [2.2 for _ in range(N)],
                [True for _ in range(N)],
                [3 + 5j for _ in range(N)],
            ]

            for scalar_list in scalar_lists: 
                # test out of place
                foreach_exeption = False
                torch_exeption = False
                tensors = op.sample_inputs(device, dtype, N)
                method = op.get_method()
                inplace = op.get_inplace()

                try:
                    ref_res = [op.ref(t, s) for t, s in zip(tensors, scalar_list)]
                except Exception:
                    torch_exeption = True

                try:
                    fe_res = method(tensors, scalar_list)
                except Exception:
                    foreach_exeption = True

                self.assertEqual(foreach_exeption, torch_exeption)

                if not torch_exeption:
                    if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                        self.assertEqual(ref_res, fe_res, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                    else:
                        self.assertEqual(ref_res, fe_res)

                # test inplace
                foreach_inplace_exeption = False
                torch_inplace_exeption = False
                try:
                    inplace(tensors, scalar_list)
                    self.assertEqual(tensors, fe_res)
                except Exception:
                    foreach_inplace_exeption = True

                try:
                    # get torch inplace reference function
                    inplace_name = op.ref_name + "_"
                    torch_inplace = getattr(torch.Tensor, inplace_name, None)

                    for t, s in zip(tensors, scalar_list):
                        torch_inplace(t, s)
                except Exception:
                    torch_inplace_exeption = True

                self.assertEqual(foreach_inplace_exeption, torch_inplace_exeption)

    #
    # Pointwise ops
    #
    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=False, include_bool=False, include_complex=False))
    def test_addcmul(self, device, dtype):
        if self.device_type == 'cpu':
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

        if self.device_type == 'cpu':
            if dtype == torch.half:
                with self.assertRaisesRegex(RuntimeError, r"\"addcdiv_cpu_out\" not implemented for \'Half\'"):
                    self._test_pointwise_op(device, dtype, torch._foreach_addcdiv,
                                            torch._foreach_addcdiv_, torch.addcdiv)
                return
        self._test_pointwise_op(device, dtype, torch._foreach_addcdiv, torch._foreach_addcdiv_, torch.addcdiv)

    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=False, include_bool=False, include_complex=False))
    def test_min_max(self, device, dtype):
        for N in N_values:
            tensors1 = self._get_test_data(device, dtype, N)
            tensors2 = self._get_test_data(device, dtype, N)

            # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
            control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                              (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype

            expected_max = [torch.max(tensors1[i].to(dtype=control_dtype),
                                      tensors2[i].to(dtype=control_dtype)).to(dtype=dtype) for i in range(N)]

            expected_min = [torch.min(tensors1[i].to(dtype=control_dtype),
                                      tensors2[i].to(dtype=control_dtype)).to(dtype=dtype) for i in range(N)]

            res_max = torch._foreach_maximum(tensors1, tensors2)
            self.assertEqual(res_max, expected_max)

            res_min = torch._foreach_minimum(tensors1, tensors2)
            self.assertEqual(res_min, expected_min)

    @dtypes(*(torch.testing.get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_max_min_float_inf_nan(self, device, dtype):
        a = [
            torch.tensor([float('inf')], device=device, dtype=dtype),
            torch.tensor([-float('inf')], device=device, dtype=dtype),
            torch.tensor([float('nan')], device=device, dtype=dtype),
            torch.tensor([float('nan')], device=device, dtype=dtype)
        ]

        b = [
            torch.tensor([-float('inf')], device=device, dtype=dtype),
            torch.tensor([float('inf')], device=device, dtype=dtype),
            torch.tensor([float('inf')], device=device, dtype=dtype),
            torch.tensor([float('nan')], device=device, dtype=dtype)
        ]

        expected = [torch.max(a1, b1) for a1, b1 in zip(a, b)]
        res = torch._foreach_maximum(a, b)
        self.assertEqual(expected, res)

        expected = [torch.min(a1, b1) for a1, b1 in zip(a, b)]
        res = torch._foreach_minimum(a, b)
        self.assertEqual(expected, res)

    @dtypes(*(torch.testing.get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_max_min_inf_nan(self, device, dtype):
        a = [
            torch.tensor([inf], device=device, dtype=dtype),
            torch.tensor([-inf], device=device, dtype=dtype),
            torch.tensor([nan], device=device, dtype=dtype),
            torch.tensor([nan], device=device, dtype=dtype)
        ]

        b = [
            torch.tensor([-inf], device=device, dtype=dtype),
            torch.tensor([inf], device=device, dtype=dtype),
            torch.tensor([inf], device=device, dtype=dtype),
            torch.tensor([nan], device=device, dtype=dtype)
        ]

        expected_max = [torch.max(a1, b1) for a1, b1 in zip(a, b)]
        res_max = torch._foreach_maximum(a, b)
        self.assertEqual(expected_max, res_max)

        expected_min = [torch.min(a1, b1) for a1, b1 in zip(a, b)]
        res_min = torch._foreach_minimum(a, b)
        self.assertEqual(expected_min, res_min)

    #
    # Special cases
    #
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
    def test_bin_op_list_error_cases(self, device):
        for bin_op, bin_op_, _ in self.bin_ops:
            tensors1 = []
            tensors2 = []

            # Empty lists
            with self.assertRaisesRegex(RuntimeError, "There were no tensor arguments to this function"):
                bin_op(tensors1, tensors2)
            with self.assertRaisesRegex(RuntimeError, "There were no tensor arguments to this function"):
                bin_op_(tensors1, tensors2)

            # One empty list
            tensors1.append(torch.tensor([1], device=device))
            with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
                bin_op(tensors1, tensors2)
            with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
                bin_op_(tensors1, tensors2)

            # Lists have different amount of tensors
            tensors2.append(torch.tensor([1], device=device))
            tensors2.append(torch.tensor([1], device=device))
            with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 1 and 2"):
                bin_op(tensors1, tensors2)
            with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 1 and 2"):
                bin_op_(tensors1, tensors2)

            # Different dtypes
            tensors1 = [torch.zeros(10, 10, device=device, dtype=torch.float) for _ in range(10)]
            tensors2 = [torch.ones(10, 10, device=device, dtype=torch.int) for _ in range(10)]

            with self.assertRaisesRegex(RuntimeError, "All tensors in the tensor list must have the same dtype."):
                bin_op(tensors1, tensors2)
            with self.assertRaisesRegex(RuntimeError, "All tensors in the tensor list must have the same dtype."):
                bin_op_(tensors1, tensors2)

            # different devices
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                tensor1 = torch.zeros(10, 10, device="cuda:0")
                tensor2 = torch.ones(10, 10, device="cuda:1")
                with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                    bin_op([tensor1], [tensor2])
                with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                    bin_op_([tensor1], [tensor2])

            # Corresponding tensors with different sizes
            tensors1 = [torch.zeros(10, 10, device=device) for _ in range(10)]
            tensors2 = [torch.ones(11, 11, device=device) for _ in range(10)]
            with self.assertRaisesRegex(RuntimeError, "Corresponding tensors in lists must have the same size"):
                bin_op(tensors1, tensors2)
            with self.assertRaisesRegex(RuntimeError, r", got \[10, 10\] and \[11, 11\]"):
                bin_op_(tensors1, tensors2)

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
            if self.device_type == 'cpu':
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    self._test_bin_op_list(device, dtype, torch._foreach_div, torch._foreach_div_, torch.div)
            else:
                self.skipTest("Skipped! See https://github.com/pytorch/pytorch/issues/44489")
            return

        for N in N_values:
            tensors1 = self._get_test_data(device, dtype, N)

            if dtype in [torch.bfloat16, torch.bool, torch.float16]:
                tensors2 = [torch.zeros(N, N, device=device, dtype=dtype).add(2) for _ in range(N)]
            else:
                tensors2 = self._get_test_data(device, dtype, N)

            expected = [torch.div(tensors1[i], tensors2[i]) for i in range(N)]
            res = torch._foreach_div(tensors1, tensors2)
            torch._foreach_div_(tensors1, tensors2)
            self.assertEqual(res, tensors1)
            self.assertEqual(tensors1, res)

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
