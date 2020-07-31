from math import pi as PI

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes
from torch.testing._internal.common_utils import TestCase, run_tests

devices = (torch.device('cpu'), torch.device('cuda:0'))


def dtype_name(dtype):
    if dtype == torch.float32:
        return 'Float'
    if dtype == torch.float64:
        return 'Double'
    if dtype == torch.complex64:
        return 'ComplexFloat'
    if dtype == torch.complex128:
        return 'ComplexDouble'
    raise NotImplementedError


class TestComplexTensor(TestCase):
    @dtypes(*torch.testing.get_all_complex_dtypes())
    def test_to_list(self, device, dtype):
        # test that the complex float tensor has expected values and
        # there's no garbage value in the resultant list
        self.assertEqual(torch.zeros((2, 2), device=device, dtype=dtype).tolist(), [[0j, 0j], [0j, 0j]])

    @dtypes(torch.float32, torch.float64)
    def test_dtype_inference(self, device, dtype):
        # issue: https://github.com/pytorch/pytorch/issues/36834
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        x = torch.tensor([3., 3. + 5.j], device=device)
        torch.set_default_dtype(default_dtype)
        self.assertEqual(x.dtype, torch.cdouble if dtype == torch.float64 else torch.cfloat)

    @dtypes(torch.float32, torch.float64)
    def test_torch_complex(self, device, dtype):
        real = torch.tensor([1, 2], device=device, dtype=dtype)
        imag = torch.tensor([3, 4], device=device, dtype=dtype)
        z = torch.complex(real, imag)
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.assertEqual(torch.tensor([1.0 + 3.0j, 2.0 + 4.0j], dtype=complex_dtype), z)

    @dtypes(torch.float32, torch.float64)
    def test_torch_complex_polar(self, device, dtype):
        abs = torch.tensor([1, 2, -3, -4.5, 1, 1], device=device, dtype=dtype)
        angle = torch.tensor([PI / 2, 5 * PI / 4, 0, -11 * PI / 6, PI, -PI],
                             device=device, dtype=dtype)
        z = torch.complex_polar(abs, angle)
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.assertEqual(torch.tensor([1j, -1.41421356237 - 1.41421356237j, -3,
                                       -3.89711431703 - 2.25j, -1, -1],
                                      dtype=complex_dtype),
                         z, atol=1e-5, rtol=1e-5)

    @dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float16, torch.complex64, torch.complex128, torch.bool)
    def test_torch_complex_floating_dtype_error(self, device, dtype):
        for op in (torch.complex, torch.complex_polar):
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=dtype)
            error = r"Expected both inputs to be Float or Double tensors but " \
                    r"got [A-Za-z]+ and [A-Za-z]+"
        with self.assertRaisesRegex(RuntimeError, error):
            op(a, b)

    @dtypes(torch.float32, torch.float64)
    def test_torch_complex_same_dtype_error(self, device, dtype):
        for op in (torch.complex, torch.complex_polar):
            other_dtype = torch.float64 if dtype == torch.float32 else torch.float32
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=other_dtype)
            error = "Expected object of scalar type {} but got scalar type " \
                    "{} for second argument".format(dtype_name(dtype),
                                                    dtype_name(other_dtype))
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b)

    @dtypes(torch.float32, torch.float64)
    def test_torch_complex_out_dtype_error(self, device, dtype):
        for op in (torch.complex, torch.complex_polar):
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=dtype)
            out = torch.zeros(2, device=device, dtype=dtype)
            expected_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
            error = "Expected object of scalar type {} but got scalar type " \
                    "{} for argument 'out'".format(dtype_name(expected_dtype),
                                                   dtype_name(dtype))
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b, out=out)

    @dtypes(torch.float32, torch.float64)
    def test_torch_complex_backward(self, device, dtype):
        real = torch.tensor([1, 2], device=device, dtype=dtype, requires_grad=True)
        imag = torch.tensor([3, 4], device=device, dtype=dtype, requires_grad=True)
        z = torch.complex(real, imag)
        loss = z.sum()
        loss.backward()
        print(real.grad)
        self.assertEqual(torch.tensor([1.0, 1.0], dtype=dtype), real.grad, atol=1e-5, rtol=1e-5)
        self.assertEqual(torch.tensor([0.0, 0.0], dtype=dtype), imag.grad, atol=1e-5, rtol=1e-5)

    @dtypes(torch.float32, torch.float64)
    def test_torch_complex_polar_backward(self, device, dtype):
        abs = torch.tensor([1, 2], device=device, dtype=dtype, requires_grad=True)
        angle = torch.tensor([PI / 2, 5 * PI / 4], device=device, dtype=dtype, requires_grad=True)
        z = torch.complex_polar(abs, angle)
        loss = z.sum()
        loss.backward()
        self.assertEqual(torch.tensor([0.0, -0.70710678118], dtype=dtype), abs.grad, atol=1e-5, rtol=1e-5)
        self.assertEqual(torch.tensor([0.0, 1.41421356237], dtype=dtype), angle.grad, atol=1e-5, rtol=1e-5)


instantiate_device_type_tests(TestComplexTensor, globals())

if __name__ == '__main__':
    run_tests()
