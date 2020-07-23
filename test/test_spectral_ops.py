
import torch
import unittest
import warnings

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_NUMPY, TEST_MKL)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, onlyOnCPUAndCUDA, precisionOverride,
     skipCPUIfNoMkl)

if TEST_NUMPY:
    import numpy as np

# Tests of functions related to Fourier analysis in the torch.fft namespace

class TestFFT(TestCase):
    exact_dtype = True

    @unittest.skipIf(not TEST_NUMPY, 'NumPy not found')
    @precisionOverride({torch.complex64: 1e-4})
    @dtypes(torch.complex64, torch.complex128)
    def test_fft(self, device, dtype):
        test_inputs = (torch.randn(67, device=device, dtype=dtype),
                       torch.randn(4029, device=device, dtype=dtype))

        def fn(t):
            return torch.fft.fft(t)
        scripted_fn = torch.jit.script(fn)

        def method_fn(t):
            return t.fft()
        scripted_method_fn = torch.jit.script(method_fn)

        torch_fns = (torch.fft.fft, torch.Tensor.fft, scripted_fn, scripted_method_fn)

        for input in test_inputs:
            expected = np.fft.fft(input.cpu().numpy())
            for fn in torch_fns:
                actual = fn(input)
                self.assertEqual(actual, expected, exact_dtype=(dtype is torch.complex128))


    # Note: NumPy will throw a ValueError for an empty input
    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    @dtypes(torch.complex64, torch.complex128)
    def test_empty_fft(self, device, dtype):
        t = torch.empty(0, device=device, dtype=dtype)

        if self.device_type == 'cuda':
            with self.assertRaisesRegex(RuntimeError, "cuFFT error"):
                torch.fft.fft(t)
            return

        # CPU (MKL)
        with self.assertRaisesRegex(RuntimeError, "MKL FFT error"):
            torch.fft.fft(t)

    @dtypes(torch.int64, torch.float32)
    def test_fft_invalid_dtypes(self, device, dtype):
        if dtype.is_floating_point:
            t = torch.randn(64, device=device, dtype=dtype)
        else:
            t = torch.randint(-2, 2, (64,), device=device, dtype=dtype)

        with self.assertRaisesRegex(RuntimeError, "Expected a complex tensor"):
            torch.fft.fft(t)

instantiate_device_type_tests(TestFFT, globals())

if __name__ == '__main__':
    run_tests()
