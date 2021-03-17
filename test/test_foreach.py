import torch
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_ROCM, TEST_WITH_SLOW
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, skipCUDAIfRocm
from torch._six import inf, nan

# Includes some values such that N * N won't be a multiple of 4,
# which should ensure we test the vectorized and non-vectorized
# kernel code paths.
N_values = [20, 23] if not TEST_WITH_SLOW else [23, 30, 300]

class TestForeach(TestCase):
    bin_ops = [
        (torch._foreach_add, torch._foreach_add_, torch.add),
        (torch._foreach_sub, torch._foreach_sub_, torch.sub),
        (torch._foreach_mul, torch._foreach_mul_, torch.mul),
        (torch._foreach_div, torch._foreach_div_, torch.div),
    ]

    unary_ops = [
        # foreach_op, foreach_op_, torch_op, bf16, complex64/128
        (torch._foreach_sqrt, torch._foreach_sqrt_, torch.sqrt, True , True),
        (torch._foreach_exp, torch._foreach_exp_, torch.exp, True, True),
        (torch._foreach_acos, torch._foreach_acos_, torch.acos, False, True),
        (torch._foreach_asin, torch._foreach_asin_, torch.asin, False, True),
        (torch._foreach_atan, torch._foreach_atan_, torch.atan, False, True),
        (torch._foreach_cos, torch._foreach_cos_, torch.cos, True, True),
        (torch._foreach_cosh, torch._foreach_cosh_, torch.cosh, False, True),
        (torch._foreach_log, torch._foreach_log_, torch.log, True, True),
        (torch._foreach_log10, torch._foreach_log10_, torch.log10, True, True),
        (torch._foreach_log2, torch._foreach_log2_, torch.log2, True, True),
        (torch._foreach_neg, torch._foreach_neg_, torch.neg, True, True),
        (torch._foreach_tan, torch._foreach_tan_, torch.tan, False, True),
        (torch._foreach_tanh, torch._foreach_tanh_, torch.tanh, True, True),
        (torch._foreach_sin, torch._foreach_sin_, torch.sin, False, True),
        (torch._foreach_sinh, torch._foreach_sinh_, torch.sinh, False, True),
        (torch._foreach_ceil, torch._foreach_ceil_, torch.ceil, False, False),
        (torch._foreach_erf, torch._foreach_erf_, torch.erf, True, False),
        (torch._foreach_erfc, torch._foreach_erfc_, torch.erfc, False, False),
        (torch._foreach_expm1, torch._foreach_expm1_, torch.expm1, False, False),
        (torch._foreach_floor, torch._foreach_floor_, torch.floor, False, False),
        (torch._foreach_log1p, torch._foreach_log1p_, torch.log1p, True, False),
        (torch._foreach_round, torch._foreach_round_, torch.round, False, False),
        (torch._foreach_frac, torch._foreach_frac_, torch.frac, False, False),
        (torch._foreach_reciprocal, torch._foreach_reciprocal_, torch.reciprocal, True, True),
        (torch._foreach_sigmoid, torch._foreach_sigmoid_, torch.sigmoid, True, False),
        (torch._foreach_trunc, torch._foreach_trunc_, torch.trunc, False, False),

        # See test_abs
        # (torch._foreach_abs, torch._foreach_abs_, torch.abs, True, True),
    ]

    def _get_test_data(self, device, dtype, N):
        if dtype in [torch.bfloat16, torch.bool, torch.float16]:
            tensors = [torch.randn(N, N, device=device).to(dtype) for _ in range(N)]
        elif dtype in torch.testing.get_all_int_dtypes():
            # Constrains the range between 1 and 10 for less stress on int8 tensors.
            tensors = [torch.randint(1, 10, (N, N), device=device, dtype=dtype) for _ in range(N)]
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
            # Constrains the range a bit for int8 tensors.
            values = [2 + (i % 5) for i in range(N)]
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
                    self.assertEqual(tensors, expected, atol=3.e-3, rtol=self.dtype_precisions[dtype][0])
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

                    msg = "Tensor lists must have the same number of tensors, got {} and {}".format(N + 1, N)

                    tensors = self._get_test_data(device, dtype, N + 1)
                    with self.assertRaisesRegex(RuntimeError, msg):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N)])

                    tensors1 = self._get_test_data(device, dtype, N + 1)
                    with self.assertRaisesRegex(RuntimeError, msg):
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

    #
    # Unary ops
    #
    @dtypes(*(torch.testing.floating_and_complex_types_and(torch.bfloat16, torch.half)))
    def test_unary_ops(self, device, dtype):
        for fe_op, fe_op_, torch_op, support_bfloat16, support_complex in self.unary_ops:
            for N in N_values:
                tensors1 = self._get_test_data(device, dtype, N)
                # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
                control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                                  (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype

                if self.device_type == 'cpu' and dtype == torch.half and torch_op not in [torch.neg, torch.frac, torch.reciprocal]:
                    with self.assertRaisesRegex(RuntimeError, r"not implemented for \'Half\'"):
                        expected = [torch_op(tensors1[i]) for i in range(N)]

                    with self.assertRaisesRegex(RuntimeError, r"not implemented for \'Half\'"):
                        res = fe_op(tensors1)
                    break

                if dtype == torch.bfloat16 and not support_bfloat16:
                    if self.device_type == 'cuda' or torch_op in [torch.sinh, torch.cosh]:
                        with self.assertRaisesRegex(RuntimeError, r"not implemented for \'BFloat16\'"):
                            expected = [torch_op(tensors1[i]) for i in range(N)]

                        with self.assertRaisesRegex(RuntimeError, r"not implemented for \'BFloat16\'"):
                            res = fe_op(tensors1)
                        break

                if dtype in [torch.complex64, torch.complex128] and not support_complex:
                    if not (self.device_type == 'cpu' and torch_op in [torch.sigmoid]):
                        # not using assertRaisesRegex due to different error messages
                        with self.assertRaises(RuntimeError):
                            expected = [torch_op(tensors1[i]) for i in range(N)]

                        with self.assertRaises(RuntimeError):
                            res = fe_op(tensors1)
                        break

                expected = [torch_op(tensors1[i].to(dtype=control_dtype)).to(dtype=dtype) for i in range(N)]
                res = fe_op(tensors1)
                if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(res, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])

                    fe_op_(tensors1)
                    self.assertEqual(res, tensors1)
                else:
                    self.assertEqual(res, expected)

                    fe_op_(tensors1)
                    self.assertEqual(res, tensors1)

    # Separate test for abs due to a lot of special cases
    # Absolute value of a complex number a + bj is defined as sqrt(a^2 + b^2), i.e. a floating point
    @dtypes(*torch.testing.get_all_dtypes())
    def test_abs(self, device, dtype):
        for N in N_values:
            tensors1 = self._get_test_data(device, dtype, N)
            # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
            control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                              (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype

            if dtype == torch.bool and self.device_type == 'cpu':
                with self.assertRaisesRegex(RuntimeError, r"not implemented for"):
                    expected = [torch.abs(tensors1[i].to(dtype=control_dtype)).to(dtype=dtype) for i in range(N)]
                continue

            expected = [torch.abs(tensors1[i].to(dtype=control_dtype)).to(dtype=dtype) for i in range(N)]
            res = torch._foreach_abs(tensors1)
            if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                self.assertEqual(res, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])

                torch._foreach_abs_(tensors1)
                self.assertEqual(res, tensors1)
            else:
                expected = [torch.abs(tensors1[i]) for i in range(N)]
                self.assertEqual(res, expected)

                if dtype in [torch.complex64, torch.complex128]:
                    with self.assertRaisesRegex(RuntimeError, r"In-place abs is not supported for complex tensors."):
                        torch._foreach_abs_(tensors1)
                else:
                    torch._foreach_abs_(tensors1)
                    self.assertEqual(res, tensors1)

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
    # Ops with scalar
    #
    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_int_scalar_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalar = 3

                if dtype == torch.bool:
                    if foreach_bin_op == torch._foreach_sub:
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                            res = foreach_bin_op(tensors, scalar)
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                            expected = [torch_bin_op(t, scalar) for t in tensors]

                        # Test In-place
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                            foreach_bin_op_(tensors, scalar)

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                            [t.sub_(scalar) for t in tensors]
                        continue

                    res = foreach_bin_op(tensors, scalar)
                    expected = [torch_bin_op(t, scalar) for t in tensors]
                    self.assertEqual(res, expected)

                    # Test In-place
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalar)

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        [t.div_(scalar) for t in tensors]

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        [t.mul_(scalar) for t in tensors]

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        [t.add_(scalar) for t in tensors]

                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool"):
                        [t.sub_(scalar) for t in tensors]
                    continue

                expected = [torch_bin_op(t, scalar) for t in tensors]
                res = foreach_bin_op(tensors, scalar)

                # In case of In-place division with integers, we can't change the dtype
                if foreach_bin_op_ == torch._foreach_div_ and dtype in torch.testing.integral_types():
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        [t.div_(scalar) for t in tensors]

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        torch._foreach_div_(tensors, scalar)
                    continue

                self.assertEqual(res, expected)

                # In case of In-place op, we can't change the dtype
                foreach_bin_op_(tensors, scalar)
                self.assertEqual(tensors, expected)

    # TODO[Fix scalar list]:
    # We need to update codegen to correctly handle function overloads with float[] and int[].
    # As optimizers work with float tensors, the result will always be torch.float32 for now.
    # Current schema is using 'float[]' as scalar list type.
    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_int_scalarlist_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalars = [1 for _ in range(N)]
                expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]

                # we dont support bool and complex types on CUDA for now
                if dtype in torch.testing.get_all_complex_dtypes() and self.device_type == 'cuda':
                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        foreach_bin_op_(tensors, scalars)

                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        foreach_bin_op(tensors, scalars)
                    return

                res = foreach_bin_op(tensors, scalars)

                if dtype == torch.bool:
                    self.assertEqual(res, [torch_bin_op(t, s) for t, s in zip(tensors, scalars)])

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalars)
                    return

                # test out of place
                self.assertEqual(res, expected)

                # test in-place
                if dtype in torch.testing.floating_types() and self.device_type == 'cpu':
                    foreach_bin_op_(tensors, scalars)
                    return
                else:
                    if foreach_bin_op_ == torch._foreach_div_ and dtype in torch.testing.integral_types():
                        with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                            foreach_bin_op_(tensors, scalars)
                    else:
                        foreach_bin_op_(tensors, scalars)
                        self.assertEqual(res, tensors)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_float_scalar_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalar = 3.3

                # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
                control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                                  (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype
                expected = [torch_bin_op(t.to(dtype=control_dtype),
                                         scalar) for t in tensors]
                if (dtype is torch.float16 or dtype is torch.bfloat16):
                    expected = [e.to(dtype=dtype) for e in expected]

                if dtype == torch.bool:
                    if foreach_bin_op == torch._foreach_sub:
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with two bool"):
                            foreach_bin_op_(tensors, scalar)

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with two bool"):
                            foreach_bin_op(tensors, scalar)
                    return

                res = foreach_bin_op(tensors, scalar)
                if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(res, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                else:
                    self.assertEqual(res, expected)

                if dtype in torch.testing.integral_types():
                    with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalar)
                    return

                foreach_bin_op_(tensors, scalar)
                if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(tensors, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                else:
                    self.assertEqual(tensors, expected)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_float_scalarlist_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalars = [1.1 for _ in range(N)]

                # Bool case
                if dtype == torch.bool:
                    if foreach_bin_op == torch._foreach_sub:
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool tensor"):
                            expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool tensor"):
                            res = foreach_bin_op(tensors, scalars)

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool tensor"):
                            [t.sub_(scalar) for t, scalar in zip(tensors, scalars)]

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool tensor"):
                            foreach_bin_op_(tensors, scalars)
                        continue

                    res = foreach_bin_op(tensors, scalars)
                    expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]
                    self.assertEqual(res, expected)

                    with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalars)
                    continue

                # If incoming dtype is float16 or bfloat16, runs in float32 and casts output back to dtype.
                control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                                  (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype
                expected = [torch_bin_op(t.to(dtype=control_dtype), s) for t, s in zip(tensors, scalars)]
                if (dtype is torch.float16 or dtype is torch.bfloat16):
                    expected = [e.to(dtype=dtype) for e in expected]

                # we dont support bool and complex types on CUDA for now
                if (dtype in torch.testing.get_all_complex_dtypes() or dtype == torch.bool) and self.device_type == 'cuda':
                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        foreach_bin_op_(tensors, scalars)

                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        foreach_bin_op(tensors, scalars)
                    return

                res = foreach_bin_op(tensors, scalars)

                if dtype in torch.testing.integral_types() and self.device_type == 'cuda':
                    self.assertEqual(res, expected)
                    with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalars)
                    continue
                else:
                    if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                        self.assertEqual(res, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                    else:
                        self.assertEqual(res, expected)

                if dtype in torch.testing.integral_types() and self.device_type == "cpu":
                    with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalars)
                    continue

                foreach_bin_op_(tensors, scalars)
                if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(tensors, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                else:
                    self.assertEqual(tensors, expected)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_complex_scalar_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalar = 3 + 5j

                # Bool case
                if dtype == torch.bool:
                    if foreach_bin_op == torch._foreach_sub:
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                            foreach_bin_op_(tensors, scalar)

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                            foreach_bin_op(tensors, scalar)
                    continue

                res = foreach_bin_op(tensors, scalar)
                expected = [torch_bin_op(t, scalar) for t in tensors]
                self.assertEqual(res, expected)

                if dtype in torch.testing.get_all_fp_dtypes() and self.device_type == 'cuda':
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalar)
                    continue

                if dtype not in [torch.complex64, torch.complex128]:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalar)
                else:
                    foreach_bin_op_(tensors, scalar)
                    self.assertEqual(res, tensors)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_complex_scalarlist_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalars = [3 + 5j for _ in range(N)]

                # Bool case
                if dtype == torch.bool:
                    if foreach_bin_op == torch._foreach_sub:
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                            foreach_bin_op_(tensors, scalars)

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                            foreach_bin_op(tensors, scalars)
                    continue

                expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]
                res = foreach_bin_op(tensors, scalars)
                self.assertEqual(res, expected)

                if dtype not in [torch.complex64, torch.complex128]:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalars)
                else:
                    foreach_bin_op_(tensors, scalars)
                    self.assertEqual(res, tensors)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_bool_scalar_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalar = True

                if foreach_bin_op == torch._foreach_sub:
                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        expected = [torch_bin_op(t, scalar) for t in tensors]

                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        foreach_bin_op(tensors, scalar)

                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        foreach_bin_op_(tensors, scalar)
                    continue

                expected = [torch_bin_op(t, scalar) for t in tensors]
                res = foreach_bin_op(tensors, scalar)
                self.assertEqual(expected, res)

                if dtype in torch.testing.integral_types_and(torch.bool) and foreach_bin_op == torch._foreach_div:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output"):
                        foreach_bin_op_(tensors, scalar)
                else:
                    foreach_bin_op_(tensors, scalar)
                    self.assertEqual(tensors, res)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_bool_scalarlist_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalars = [True for _ in range(N)]

                # we dont support complex types on CUDA for now
                if (dtype in torch.testing.get_all_complex_dtypes()) and self.device_type == 'cuda':
                    # There are a two types of different errors that will be thrown.
                    # - Not implemented
                    # - Subtraction with a bool tensor
                    with self.assertRaises(RuntimeError):
                        foreach_bin_op_(tensors, scalars)

                    with self.assertRaises(RuntimeError):
                        foreach_bin_op(tensors, scalars)
                    continue

                if foreach_bin_op == torch._foreach_sub:
                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]

                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        foreach_bin_op(tensors, scalars)

                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        foreach_bin_op_(tensors, scalars)
                    continue

                expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]
                res = foreach_bin_op(tensors, scalars)
                self.assertEqual(expected, res)

                if dtype in torch.testing.integral_types_and(torch.bool) and foreach_bin_op == torch._foreach_div:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output"):
                        foreach_bin_op_(tensors, scalars)
                else:
                    foreach_bin_op_(tensors, scalars)
                    self.assertEqual(tensors, res)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_mixed_scalarlist_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalars = [1, 1.1, 3 + 5j] + [True for _ in range(N - 3)]

                if foreach_bin_op == torch._foreach_sub:
                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]

                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        foreach_bin_op(tensors, scalars)

                    # There are a two types of different errors that will be thrown.
                    # - Sub with bool is not allowed.
                    # - Result type can't be cast to the desired output type
                    with self.assertRaises(RuntimeError):
                        foreach_bin_op_(tensors, scalars)
                    continue

                expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]
                res = foreach_bin_op(tensors, scalars)
                self.assertEqual(expected, res)

                if dtype in torch.testing.get_all_complex_dtypes():
                    foreach_bin_op_(tensors, scalars)
                    self.assertEqual(expected, tensors)
                else:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalars)

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
