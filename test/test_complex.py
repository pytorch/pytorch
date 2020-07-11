import numpy as np

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes
from torch.testing._internal.common_utils import TestCase, run_tests

devices = (torch.device('cpu'), torch.device('cuda:0'))


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
        real = torch.tensor([1, 2], device=device, dtype=torch.int32)
        imag = torch.tensor([3, 4], device=device, dtype=dtype)
        z = torch.complex(real, imag)
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.assertEqual(torch.tensor([1.0 + 3.0j, 2.0 + 4.0j], dtype=complex_dtype), z)

    @dtypes(torch.float32, torch.float64)
    def test_torch_complex_polar(self, device, dtype):
        abs = torch.tensor([1, 2], device=device, dtype=torch.int32)
        angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], device=device, dtype=dtype)
        z = torch.complex_polar(abs, angle)
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.assertEqual(torch.tensor([0 + 1.0j, -1.41421356237 - 1.41421356237j],
                                      dtype=complex_dtype),
                         z, atol=1e-5, rtol=1e-5)

    @dtypes(torch.int32, torch.int64, torch.complex64, torch.complex128)
    def test_torch_complex_error(self, device, dtype):
        abs = torch.tensor([1, 2], device=device, dtype=dtype)
        angle = torch.tensor([3, 4], device=device, dtype=dtype)
        if device.startswith('cuda'):
            error = r"\"complex_polar_cuda\" not implemented for '[A-Za-z]+'"
        else:
            error = r"\"complex_polar_cpu\" not implemented for '[A-Za-z]+'"
        with self.assertRaisesRegex(RuntimeError, error):
            torch.complex_polar(abs, angle)

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
        angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], device=device, dtype=dtype, requires_grad=True)
        z = torch.complex_polar(abs, angle)
        loss = z.sum()
        loss.backward()
        self.assertEqual(torch.tensor([0.0, -0.70710678118], dtype=dtype), abs.grad, atol=1e-5, rtol=1e-5)
        self.assertEqual(torch.tensor([0.0, 1.41421356237], dtype=dtype), angle.grad, atol=1e-5, rtol=1e-5)


instantiate_device_type_tests(TestComplexTensor, globals())

if __name__ == '__main__':
    run_tests()
