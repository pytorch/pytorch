import torch
import unittest
import itertools

from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_ROCM, TEST_WITH_SLOW
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, skipCUDAIfRocm
from torch._six import inf, nan

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

    #
    # Unary ops
    #
    @dtypes(*torch.testing.get_all_dtypes())
    def test_neg(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            # Negation, the `-` operator, on a bool tensor is not supported.
            if dtype == torch.bool:
                with self.assertRaisesRegex(RuntimeError, "Negation, the `-` operator, on a bool tensor is not supported"):
                    expected = [torch.neg(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "Negation, the `-` operator, on a bool tensor is not supported"):
                    res = torch._foreach_neg(tensors)

                with self.assertRaisesRegex(RuntimeError, "Negation, the `-` operator, on a bool tensor is not supported"):
                    torch._foreach_neg_(tensors)

                continue

            expected = [torch.neg(tensors[i]) for i in range(N)]
            res = torch._foreach_neg(tensors)
            torch._foreach_neg_(tensors)
            self.assertEqual(res, expected)
            self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_sqrt(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.sqrt(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_sqrt(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_sqrt_(tensors)
                continue

            expected = [torch.sqrt(tensors[i]) for i in range(N)]
            res = torch._foreach_sqrt(tensors)
            self.assertEqual(res, expected)

            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].sqrt_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_sqrt_(tensors)
            else:
                torch._foreach_sqrt_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_exp(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in torch.testing.integral_types_and(torch.bool) or \
               dtype in [torch.half] and self.device_type == 'cpu':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.exp(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_exp(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_exp_(tensors)
                continue

            expected = [torch.exp(tensors[i]) for i in range(N)]
            res = torch._foreach_exp(tensors)
            torch._foreach_exp_(tensors)
            self.assertEqual(res, expected)
            self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_acos(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.bfloat16] and self.device_type == 'cuda':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.acos(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_acos(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_acos_(tensors)
                continue

            # out of place
            expected = [torch.acos(tensors[i]) for i in range(N)]
            res = torch._foreach_acos(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].acos_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_acos_(tensors)
            else:
                torch._foreach_acos_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_asin(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            # if dtype in torch.testing.integral_types_and(torch.bool) or \
            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.bfloat16] and self.device_type == 'cuda':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.asin(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_asin(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_asin_(tensors)
                continue

            # out of place
            expected = [torch.asin(tensors[i]) for i in range(N)]
            res = torch._foreach_asin(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].asin_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_asin_(tensors)
            else:
                torch._foreach_asin_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_atan(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.bfloat16] and self.device_type == 'cuda':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.atan(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_atan(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_atan_(tensors)
                continue

            # out of place
            expected = [torch.atan(tensors[i]) for i in range(N)]
            res = torch._foreach_atan(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].atan_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_atan_(tensors)
            else:
                torch._foreach_atan_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_cosh(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.bfloat16]:
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.cosh(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_cosh(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_cosh_(tensors)
                continue

            # out of place
            expected = [torch.cosh(tensors[i]) for i in range(N)]
            res = torch._foreach_cosh(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].cosh_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_cosh_(tensors)
            else:
                torch._foreach_cosh_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_sin(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.bfloat16] and self.device_type == 'cuda':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.sin(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_sin(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_sin_(tensors)
                continue

            # out of place
            expected = [torch.sin(tensors[i]) for i in range(N)]
            res = torch._foreach_sin(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].sin_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_sin_(tensors)
            else:
                torch._foreach_sin_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_sinh(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.bfloat16]:
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.sinh(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_sinh(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_sinh_(tensors)
                continue

            # out of place
            expected = [torch.sinh(tensors[i]) for i in range(N)]
            res = torch._foreach_sinh(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].sinh_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_sinh_(tensors)
            else:
                torch._foreach_sinh_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_tan(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.bfloat16] and self.device_type == 'cuda':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.tan(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_tan(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_tan_(tensors)
                continue

            # out of place
            expected = [torch.tan(tensors[i]) for i in range(N)]
            res = torch._foreach_tan(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].tan_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_tan_(tensors)
            else:
                torch._foreach_tan_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_cos(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.cos(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_cos(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_cos_(tensors)
                continue

            # out of place
            expected = [torch.cos(tensors[i]) for i in range(N)]
            res = torch._foreach_cos(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].cos_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_cos_(tensors)
            else:
                torch._foreach_cos_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_log(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.log(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_log(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_log_(tensors)
                continue

            # out of place
            expected = [torch.log(tensors[i]) for i in range(N)]
            res = torch._foreach_log(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].log_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_log_(tensors)
            else:
                torch._foreach_log_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_log10(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.log10(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_log10(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_log10_(tensors)
                continue

            # out of place
            expected = [torch.log10(tensors[i]) for i in range(N)]
            res = torch._foreach_log10(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].log10_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_log10_(tensors)
            else:
                torch._foreach_log10_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_log2(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.log2(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_log2(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_log2_(tensors)
                continue

            # out of place
            expected = [torch.log2(tensors[i]) for i in range(N)]
            res = torch._foreach_log2(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].log2_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_log2_(tensors)
            else:
                torch._foreach_log2_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_tanh(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.tanh(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_tanh(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_tanh_(tensors)
                continue

            # out of place
            expected = [torch.tanh(tensors[i]) for i in range(N)]
            res = torch._foreach_tanh(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].tanh_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_tanh_(tensors)
            else:
                torch._foreach_tanh_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_ceil(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            # complex
            if dtype in [torch.complex64, torch.complex128]:
                with self.assertRaisesRegex(RuntimeError, "ceil is not supported for complex inputs"):
                    expected = [torch.ceil(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "supported for complex inputs"):
                    torch._foreach_ceil(tensors)

                with self.assertRaisesRegex(RuntimeError, "supported for complex inputs"):
                    torch._foreach_ceil_(tensors)
                continue

            # half and float16
            if dtype in [torch.bfloat16] and self.device_type == 'cuda' or \
               dtype in [torch.float16] and self.device_type == 'cpu':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.ceil(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_ceil(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_ceil_(tensors)
                continue

            # integral + bool
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.ceil(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_ceil(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_ceil_(tensors)
                continue

            # out of place
            expected = [torch.ceil(tensors[i]) for i in range(N)]
            res = torch._foreach_ceil(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].ceil_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_ceil_(tensors)
            else:
                torch._foreach_ceil_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_erf(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.complex64, torch.complex128]:
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.erf(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_erf(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_erf_(tensors)
                continue

            # out of place
            expected = [torch.erf(tensors[i]) for i in range(N)]
            res = torch._foreach_erf(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].erf_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_erf_(tensors)
            else:
                torch._foreach_erf_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_erfc(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.complex64, torch.complex128] or \
               dtype in [torch.bfloat16] and self.device_type == 'cuda':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.erfc(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_erfc(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_erfc_(tensors)
                continue

            # out of place
            expected = [torch.erfc(tensors[i]) for i in range(N)]
            res = torch._foreach_erfc(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].erfc_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_erfc_(tensors)
            else:
                torch._foreach_erfc_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_expm1(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.complex64, torch.complex128] or \
               dtype in [torch.bfloat16] and self.device_type == 'cuda':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.expm1(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_expm1(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_expm1_(tensors)
                continue

            # out of place
            expected = [torch.expm1(tensors[i]) for i in range(N)]
            res = torch._foreach_expm1(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].expm1_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_expm1_(tensors)
            else:
                torch._foreach_expm1_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_floor(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            # complex
            if dtype in [torch.complex64, torch.complex128]:
                with self.assertRaisesRegex(RuntimeError, "floor is not supported for complex inputs"):
                    expected = [torch.floor(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "supported for complex inputs"):
                    torch._foreach_floor(tensors)

                with self.assertRaisesRegex(RuntimeError, "supported for complex inputs"):
                    torch._foreach_floor_(tensors)
                continue

            # half, bfloat16, integral + bool 
            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.bfloat16] and self.device_type == 'cuda' or \
               dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.floor(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_floor(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_floor_(tensors)
                continue

            # out of place
            expected = [torch.floor(tensors[i]) for i in range(N)]
            res = torch._foreach_floor(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].floor_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_floor_(tensors)
            else:
                torch._foreach_floor_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_log1p(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.complex64, torch.complex128]:
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.log1p(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_log1p(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_log1p_(tensors)
                continue

            # out of place
            expected = [torch.log1p(tensors[i]) for i in range(N)]
            res = torch._foreach_log1p(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].log1p_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_log1p_(tensors)
            else:
                torch._foreach_log1p_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_round(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.bfloat16] and self.device_type == 'cuda' or \
               dtype in [torch.complex64, torch.complex128] or \
               dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.round(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_round(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_round_(tensors)
                continue

            # out of place
            expected = [torch.round(tensors[i]) for i in range(N)]
            res = torch._foreach_round(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].round_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_round_(tensors)
            else:
                torch._foreach_round_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_frac(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.bfloat16] and self.device_type == 'cuda' or \
               dtype in [torch.complex64, torch.complex128] or \
               dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.frac(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_frac(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_frac_(tensors)
                continue

            # out of place
            expected = [torch.frac(tensors[i]) for i in range(N)]
            res = torch._foreach_frac(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].frac_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_frac_(tensors)
            else:
                torch._foreach_frac_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_reciprocal(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.reciprocal(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_reciprocal(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_reciprocal_(tensors)
                continue

            # out of place
            expected = [torch.reciprocal(tensors[i]) for i in range(N)]
            res = torch._foreach_reciprocal(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].reciprocal_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_reciprocal_(tensors)
            else:
                torch._foreach_reciprocal_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_sigmoid(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            if dtype in [torch.half] and self.device_type == 'cpu' or \
               dtype in [torch.complex64, torch.complex128] and self.device_type == 'cuda':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.sigmoid(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_sigmoid(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_sigmoid_(tensors)
                continue

            # out of place
            expected = [torch.sigmoid(tensors[i]) for i in range(N)]
            res = torch._foreach_sigmoid(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].sigmoid_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_sigmoid_(tensors)
            else:
                torch._foreach_sigmoid_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_trunc(self, device, dtype):
        for N in N_values:
            tensors = self._get_test_data(device, dtype, N)

            # complex
            if dtype in [torch.complex64, torch.complex128]:
                with self.assertRaisesRegex(RuntimeError, "trunc is not supported for complex inputs"):
                    expected = [torch.trunc(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "supported for complex inputs"):
                    torch._foreach_trunc(tensors)

                with self.assertRaisesRegex(RuntimeError, "supported for complex inputs"):
                    torch._foreach_trunc_(tensors)
                continue

            # float16, bfloat16, integral + bool
            if dtype in [torch.float16] and self.device_type == 'cpu' or \
               dtype in [torch.bfloat16] and self.device_type == 'cuda' or \
               dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.trunc(tensors[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_trunc(tensors)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_trunc_(tensors)
                continue

            # out of place
            expected = [torch.trunc(tensors[i]) for i in range(N)]
            res = torch._foreach_trunc(tensors)
            self.assertEqual(res, expected)

            # In-place
            if dtype in torch.testing.integral_types_and(torch.bool):
                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    [tensors[i].trunc_() for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                    torch._foreach_trunc_(tensors)
            else:
                torch._foreach_trunc_(tensors)
                self.assertEqual(tensors, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_abs(self, device, dtype):
        for N in N_values:
            tensors1 = self._get_test_data(device, dtype, N)

            if dtype == torch.bool and self.device_type == 'cpu':
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    expected = [torch.abs(tensors1[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_abs(tensors1)

                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    torch._foreach_abs_(tensors1)
                continue

            expected = [torch.abs(tensors1[i]) for i in range(N)]
            res = torch._foreach_abs(tensors1)
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
    @dtypes(*torch.testing.get_all_dtypes())
    def test_addcmul(self, device, dtype):
        for N in N_values:
            # There can be a single scalar or a list of scalars
            values = [2 + i for i in range(N)]
            for vals in [values[0], values]:
                tensors = self._get_test_data(device, dtype, N)
                tensors1 = self._get_test_data(device, dtype, N)
                tensors2 = self._get_test_data(device, dtype, N)

                # Not implemented
                if dtype in [torch.half, torch.bool] and self.device_type == 'cpu' or \
                   dtype in [torch.bfloat16] and self.device_type == 'cpu' or \
                   dtype in [torch.bool] and self.device_type == 'cuda':
                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        if not isinstance(vals, list):
                            expected = [torch.addcmul(tensors[i], tensors1[i], tensors2[i], value=values[0]) for i in range(N)]
                        else:
                            expected = [torch.addcmul(tensors[i], tensors1[i], tensors2[i], value=values[i]) for i in range(N)]

                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        torch._foreach_addcmul(tensors, tensors1, tensors2, vals)

                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        torch._foreach_addcmul_(tensors, tensors1, tensors2, vals)
                    continue

                # Mimics cuda kernel dtype flow. With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
                control_dtype = torch.float32 if (self.device_type == 'cuda' and 
                                                  (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype

                if not isinstance(vals, list):
                    expected = [torch.addcmul(tensors[i].to(dtype=control_dtype),
                                              tensors1[i].to(dtype=control_dtype),
                                              tensors2[i].to(dtype=control_dtype),
                                              value=values[0]).to(dtype=dtype) for i in range(N)]
                else:
                    expected = [torch.addcmul(tensors[i].to(dtype=control_dtype),
                                              tensors1[i].to(dtype=control_dtype),
                                              tensors2[i].to(dtype=control_dtype),
                                              value=values[i]).to(dtype=dtype) for i in range(N)]

                res = torch._foreach_addcmul(tensors, tensors1, tensors2, vals)
                torch._foreach_addcmul_(tensors, tensors1, tensors2, vals)
                self.assertEqual(res, tensors)
                self.assertEqual(tensors, expected)

                # test error cases
                for op in [torch._foreach_addcmul, torch._foreach_addcmul_]:
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

    @dtypes(*torch.testing.get_all_dtypes())
    def test_addcdiv(self, device, dtype):
        for N in N_values:
            # There can be a single scalar or a list of scalars
            values = [2 + i for i in range(N)]
            for vals in [values[0], values]:
                tensors = self._get_test_data(device, dtype, N)
                tensors1 = self._get_test_data(device, dtype, N)
                tensors2 = self._get_test_data(device, dtype, N)

                # Integer division not supported
                if dtype in torch.testing.integral_types_and(torch.bool):
                    with self.assertRaisesRegex(RuntimeError, "Integer division with addcdiv is no longer supported"):
                        if not isinstance(vals, list):
                            expected = [torch.addcdiv(tensors[i], 
                                                      tensors1[i], 
                                                      tensors2[i], value=values[0]) for i in range(N)]
                        else:
                            expected = [torch.addcdiv(tensors[i], 
                                                      tensors1[i], 
                                                      tensors2[i], value=values[i]) for i in range(N)]

                    with self.assertRaisesRegex(RuntimeError, "Integer division with addcdiv is no longer supported"):
                        torch._foreach_addcdiv(tensors, tensors1, tensors2, vals)

                    with self.assertRaisesRegex(RuntimeError, "Integer division with addcdiv is no longer supported"):
                        torch._foreach_addcdiv_(tensors, tensors1, tensors2, vals)
                    continue

                # Not implemented
                if dtype == torch.half and self.device_type == 'cpu':
                    with self.assertRaisesRegex(RuntimeError, r"\"addcdiv_cpu_out\" not implemented for \'Half\'"):
                        if not isinstance(vals, list):
                            expected = [torch.addcdiv(tensors[i], tensors1[i], tensors2[i], value=values[0]) for i in range(N)]
                        else:
                            expected = [torch.addcdiv(tensors[i], tensors1[i], tensors2[i], value=values[i]) for i in range(N)]

                        with self.assertRaisesRegex(RuntimeError, r"\"addcdiv_cpu_out\" not implemented for \'Half\'"):
                            torch._foreach_addcdiv(tensors, tensors1, tensors2, vals)

                        with self.assertRaisesRegex(RuntimeError, r"\"addcdiv_cpu_out\" not implemented for \'Half\'"):
                            torch._foreach_addcdiv_(tensors, tensors1, tensors2, vals)

                # Not implemented
                if dtype in [torch.half, torch.bool] and self.device_type == 'cpu' or \
                   dtype in [torch.bfloat16] and self.device_type == 'cpu' or \
                   dtype in [torch.bool] and self.device_type == 'cuda':
                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        if not isinstance(vals, list):
                            expected = [torch.addcdiv(tensors[i], tensors1[i], tensors2[i], value=values[0]) for i in range(N)]
                        else:
                            expected = [torch.addcdiv(tensors[i], tensors1[i], tensors2[i], value=values[i]) for i in range(N)]

                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        torch._foreach_addcdiv(tensors, tensors1, tensors2, vals)

                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        torch._foreach_addcdiv_(tensors, tensors1, tensors2, vals)
                    continue

                # Mimics cuda kernel dtype flow. With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
                control_dtype = torch.float32 if (self.device_type == 'cuda' and 
                                                  (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype

                if not isinstance(vals, list):
                    expected = [torch.addcdiv(tensors[i].to(dtype=control_dtype),
                                              tensors1[i].to(dtype=control_dtype),
                                              tensors2[i].to(dtype=control_dtype),
                                              value=values[0]).to(dtype=dtype) for i in range(N)]
                else:
                    expected = [torch.addcdiv(tensors[i].to(dtype=control_dtype),
                                              tensors1[i].to(dtype=control_dtype),
                                              tensors2[i].to(dtype=control_dtype),
                                              value=values[i]).to(dtype=dtype) for i in range(N)]

                res = torch._foreach_addcdiv(tensors, tensors1, tensors2, vals)
                torch._foreach_addcdiv_(tensors, tensors1, tensors2, vals)
                self.assertEqual(res, tensors)
                self.assertEqual(tensors, expected)

                # test error cases
                for op in [torch._foreach_addcdiv, torch._foreach_addcdiv_]:
                    tensors = self._get_test_data(device, dtype, N)
                    tensors1 = self._get_test_data(device, dtype, N)
                    tensors2 = self._get_test_data(device, dtype, N)

                    with self.assertRaisesRegex(RuntimeError, 
                                                "Tensor list must have same number of elements as scalar list."):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N + 1)])

                    with self.assertRaisesRegex(RuntimeError,
                                                "Tensor list must have same number of elements as scalar list."):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N - 1)])

                    tensors = self._get_test_data(device, dtype, N + 1)
                    with self.assertRaisesRegex(RuntimeError, 
                                                "Tensor lists must have the same number of tensors, got 21 and 20"):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N)])

                    tensors1 = self._get_test_data(device, dtype, N + 1)
                    with self.assertRaisesRegex(RuntimeError, 
                                                "Tensor lists must have the same number of tensors, got 21 and 20"):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N)])

    @dtypes(*torch.testing.get_all_dtypes())
    def test_min_max(self, device, dtype):
        for N in N_values:
            tensors1 = self._get_test_data(device, dtype, N)
            tensors2 = self._get_test_data(device, dtype, N)

            if dtype in [torch.complex64, torch.complex128]:
                with self.assertRaisesRegex(RuntimeError, "maximum does not support complex inputs."):
                    [torch.max(tensors1[i], tensors2[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "minimum does not support complex inputs."):
                    [torch.min(tensors1[i], tensors2[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "foreach_maximum/foreach_minimum is not supported for complex inputs"):
                    torch._foreach_maximum(tensors1, tensors2)

                with self.assertRaisesRegex(RuntimeError, "foreach_maximum/foreach_minimum is not supported for complex inputs"):
                    torch._foreach_minimum(tensors1, tensors2)

                continue

            expected_max = [torch.max(tensors1[i], tensors2[i]) for i in range(N)]
            res_max = torch._foreach_maximum(tensors1, tensors2)
            self.assertEqual(res_max, expected_max)

            expected_min = [torch.min(tensors1[i], tensors2[i]) for i in range(N)]
            res_min = torch._foreach_minimum(tensors1, tensors2)
            self.assertEqual(res_min, expected_min)

    @dtypes(*(torch.testing.get_all_fp_dtypes()))
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

    @dtypes(*(torch.testing.get_all_fp_dtypes()))
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
    def test_int_scalar(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalar = 3

                if dtype == torch.bool and foreach_bin_op == foreach_bin_op == torch._foreach_sub:
                    if foreach_bin_op == foreach_bin_op == torch._foreach_sub:
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

                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool tensor is not supported."):
                        [t.sub_(scalar) for t in tensors]
                    continue

                expected = [torch_bin_op(t, scalar) for t in tensors]
                res = foreach_bin_op(tensors, scalar)

                # In case of In-place division with integers, we can't change the dtype
                if foreach_bin_op_ == torch._foreach_div_ and dtype in torch.testing.integral_types() and self.device_type == "cpu":
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        [t.div_(scalar) for t in tensors]

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        torch._foreach_div_(tensors, scalar)
                    continue

                self.assertEqual(res, expected)

                # In case of In-place op, we can't change the dtype
                if (expected[0].dtype == dtype):
                    foreach_bin_op_(tensors, scalar)
                    self.assertEqual(tensors, expected)
                else:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalar)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    def test_int_scalarlist(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalars = [1 for _ in range(N)]

                # special case around bool
                if dtype == torch.bool:
                    # out of place
                    if foreach_bin_op == torch._foreach_sub:
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool"): 
                            expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool"): 
                            res = foreach_bin_op(tensors, scalars)

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool"): 
                            [t.sub_(scalar) for t, scalar in zip(tensors, scalars)]

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool"): 
                            foreach_bin_op_(tensors, scalars)
                        continue
                    else:
                        expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]
                        res = foreach_bin_op(tensors, scalars)
                        self.assertEqual(res, expected)

                    # test In-place
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        [t.div_(scalar) for t, scalar in zip(tensors, scalars)]

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalars)
                    continue

                # out of place
                expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]
                res = foreach_bin_op(tensors, scalars)
                self.assertEqual(res, expected)

                # in-place
                if foreach_bin_op_ == torch._foreach_div_ and dtype in torch.testing.integral_types():
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        [t.div_(scalar) for t, scalar in zip(tensors, scalars)]

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        torch._foreach_div_(tensors, scalars)
                else:
                    foreach_bin_op_(tensors, scalars)
                    self.assertEqual(tensors, res)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    def test_float_scalar(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalar = 3.3

                # Bool case
                if dtype == torch.bool:
                    if foreach_bin_op == torch._foreach_sub:
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                            foreach_bin_op_(tensors, scalar)

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                            foreach_bin_op(tensors, scalar)
                    continue

                # Mimics cuda kernel dtype flow. With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
                control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                                  (dtype in [torch.float16, torch.bfloat16])) else dtype
                expected = [torch_bin_op(t.to(dtype=control_dtype), scalar) for t in tensors]
                if (dtype in [torch.float16, torch.bfloat16]):
                    expected = [e.to(dtype=dtype) for e in expected]

                # test out of place
                res = foreach_bin_op(tensors, scalar)
                if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(res, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                else:
                    self.assertEqual(res, expected)

                # test In-place
                if dtype in torch.testing.integral_types():
                    with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalar)
                    continue

                foreach_bin_op_(tensors, scalar)
                if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(tensors, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                else:
                    self.assertEqual(tensors, expected)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    def test_float_scalarlist(self, device, dtype):
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
    def test_complex_scalar(self, device, dtype):
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

    @dtypes(*torch.testing.get_all_dtypes())
    def test_complex_scalarlist(self, device, dtype):
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
    def test_bool_scalar(self, device, dtype):
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

                if dtype in torch.testing.integral_types_and(torch.bool) and foreach_bin_op == foreach_bin_op == torch._foreach_div:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output"):
                        foreach_bin_op_(tensors, scalar)
                else:
                    foreach_bin_op_(tensors, scalar)
                    self.assertEqual(tensors, res)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    def test_bool_scalarlist(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalars = [True for _ in range(N)]

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

                if dtype in torch.testing.integral_types_and(torch.bool) and foreach_bin_op == foreach_bin_op == torch._foreach_div:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output"):
                        foreach_bin_op_(tensors, scalars)
                else:
                    foreach_bin_op_(tensors, scalars)
                    self.assertEqual(tensors, res)

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
        for tensors in [[torch.randn([0])]]:
            res = torch._foreach_add(tensors, 1)
            self.assertEqual(res, tensors)

            torch._foreach_add_(tensors, 1)
            self.assertEqual(res, tensors)

        with self.assertRaisesRegex(RuntimeError, "There were no tensor arguments to this function"):
            torch._foreach_add([], 1)

        with self.assertRaisesRegex(RuntimeError, "There were no tensor arguments to this function"):
            torch._foreach_add_([], 1)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_overlapping_tensors(self, device, dtype):
        tensors = [torch.ones(1, 1, device=device, dtype=dtype).expand(2, 1, 3)]
        expected = [torch.tensor([[[2, 2, 2]], [[2, 2, 2]]], dtype=dtype, device=device)]

        # bool tensor + 1 will result in int64 tensor
        if dtype == torch.bool:
            expected[0] = expected[0].to(torch.int64).add(1)

        res = torch._foreach_add(tensors, 1)
        self.assertEqual(res, expected)

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

    @dtypes(*tuple(itertools.combinations_with_replacement(torch.testing.get_all_dtypes(), 2)))
    def test_add_list(self, device, dtypes):
        for N in N_values:
            tensors1 = self._get_test_data(device, dtypes[0], N)
            tensors2 = self._get_test_data(device, dtypes[1], N)
            expected = [torch.add(tensors1[i], tensors2[i]) for i in range(N)]
            res = torch._foreach_add(tensors1, tensors2)
            self.assertEqual(res, expected)

            if dtypes[0] == dtypes[1]:
                torch._foreach_add_(tensors1, tensors2)
                self.assertEqual(res, tensors1)
            else:
                if dtypes[0] not in [torch.complex32, torch.complex64, torch.complex128] and \
                   dtypes[1] in [torch.complex32, torch.complex64, torch.complex128]:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        torch._foreach_add_(tensors1, tensors2)

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        expected_ = [torch.clone(tensors1[i]).add_(tensors2[i]) for i in range(N)]

                elif dtypes[0] in torch.testing.integral_types() and dtypes[1] in torch.testing.floating_types_and(torch.bfloat16, torch.float16):
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        torch._foreach_add_(tensors1, tensors2)

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        expected_ = [torch.clone(tensors1[i]).add_(tensors2[i]) for i in range(N)]
                else:
                    expected_ = [torch.clone(tensors1[i]).add_(tensors2[i]) for i in range(N)]
                    torch._foreach_add_(tensors1, tensors2)
                    self.assertEqual(expected_, tensors1)

    @dtypes(*tuple(itertools.combinations_with_replacement(torch.testing.get_all_dtypes(), 2)))
    def test_sub_list(self, device, dtypes):
        for N in N_values:
            tensors1 = self._get_test_data(device, dtypes[0], N)
            tensors2 = self._get_test_data(device, dtypes[1], N)
            if dtypes[0] == torch.bool or dtypes[1] == torch.bool:
                with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                    expected = [torch.sub(tensors1[i], tensors2[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                    res = torch._foreach_sub(tensors1, tensors2)
            else:
                if dtypes[0] not in [torch.complex32, torch.complex64, torch.complex128] and \
                   dtypes[1] in [torch.complex32, torch.complex64, torch.complex128]:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        torch._foreach_sub_(tensors1, tensors2)

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        expected_ = [torch.clone(tensors1[i]).sub_(tensors2[i]) for i in range(N)]

                elif dtypes[0] in torch.testing.integral_types() and dtypes[1] in torch.testing.floating_types_and(torch.bfloat16, torch.float16):
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        torch._foreach_sub_(tensors1, tensors2)

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        expected_ = [torch.clone(tensors1[i]).sub_(tensors2[i]) for i in range(N)]
                else:
                    expected_ = [torch.clone(tensors1[i]).sub_(tensors2[i]) for i in range(N)]
                    torch._foreach_sub_(tensors1, tensors2)
                    self.assertEqual(expected_, tensors1)

    @dtypes(*tuple(itertools.combinations_with_replacement(torch.testing.get_all_dtypes(), 2)))
    def test_mul_list(self, device, dtypes):
        for N in N_values:
            tensors1 = self._get_test_data(device, dtypes[0], N)
            tensors2 = self._get_test_data(device, dtypes[1], N)
            expected = [torch.mul(tensors1[i], tensors2[i]) for i in range(N)]
            res = torch._foreach_mul(tensors1, tensors2)
            self.assertEqual(res, expected)

            if dtypes[0] not in [torch.complex32, torch.complex64, torch.complex128] and \
               dtypes[1] in [torch.complex32, torch.complex64, torch.complex128]:
                with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                    torch._foreach_mul_(tensors1, tensors2)

                with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                    expected_ = [torch.clone(tensors1[i]).mul_(tensors2[i]) for i in range(N)]
            elif dtypes[0] in torch.testing.integral_types() and dtypes[1] in torch.testing.floating_types_and(torch.bfloat16, torch.float16):
                with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                    torch._foreach_mul_(tensors1, tensors2)

                with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                    expected_ = [torch.clone(tensors1[i]).mul_(tensors2[i]) for i in range(N)]
            else:
                expected_ = [torch.clone(tensors1[i]).mul_(tensors2[i]) for i in range(N)]
                torch._foreach_mul_(tensors1, tensors2)
                self.assertEqual(expected_, tensors1)

    @dtypes(*tuple(itertools.combinations_with_replacement(torch.testing.get_all_dtypes(), 2)))
    def test_div_list(self, device, dtypes):
        for N in N_values:
            tensors1 = self._get_test_data(device, dtypes[0], N)
            tensors2 = self._get_test_data(device, dtypes[1], N)

            if dtypes[0] in torch.testing.integral_types_and(torch.bool):
                # In case of In-place division with integers, we can't change the dtype
                with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                    expected_ = [torch.clone(tensors1[i]).div_(tensors2[i]) for i in range(N)]

                with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                    torch._foreach_div_(tensors1, tensors2)

                continue

            expected = [torch.div(tensors1[i], tensors2[i]) for i in range(N)]
            res = torch._foreach_div(tensors1, tensors2)
            self.assertEqual(res, expected)

            if dtypes[0] not in [torch.complex32, torch.complex64, torch.complex128] and \
               dtypes[1] in [torch.complex32, torch.complex64, torch.complex128]:
                with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                    torch._foreach_div_(tensors1, tensors2)

                with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                    expected_ = [torch.clone(tensors1[i]).div_(tensors2[i]) for i in range(N)]
            elif dtypes[0] in torch.testing.integral_types() and dtypes[1] in torch.testing.floating_types_and(torch.bfloat16, torch.float16):
                with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                    torch._foreach_div_(tensors1, tensors2)

                with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                    expected_ = [torch.clone(tensors1[i]).div_(tensors2[i]) for i in range(N)]
            else:
                expected_ = [torch.clone(tensors1[i]).div_(tensors2[i]) for i in range(N)]
                torch._foreach_div_(tensors1, tensors2)
                self.assertEqual(expected_, tensors1)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_list_different_sizes(self, device, dtype):
        tensors1 = [torch.zeros(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]
        tensors2 = [torch.ones(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]
        expected = [torch.add(a, b) for a, b in zip(tensors1, tensors2)]

        res = torch._foreach_add(tensors1, tensors2)
        self.assertEqual(res, expected)

        torch._foreach_add_(tensors1, tensors2)
        self.assertEqual(expected, tensors1)

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