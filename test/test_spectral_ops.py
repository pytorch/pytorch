
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
        test_inputs = (torch.randn(1029, device=device, dtype=dtype),)

        for input in test_inputs:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.compare_with_numpy(torch.fft.fft,
                                        np.fft.fft,
                                        input,
                                        exact_dtype=(dtype is torch.complex128))

                if dtype is torch.complex64:
                    self.assertEqual(len(w), 1)

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


instantiate_device_type_tests(TestFFT, globals())

if __name__ == '__main__':
    run_tests()
