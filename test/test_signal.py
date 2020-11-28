import torch

import math

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, slowTest, TEST_SCIPY)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, onlyCPU, onlyOnCPUAndCUDA, dtypes)

class TestSignal(TestCase):
    @onlyOnCPUAndCUDA
    @precisionOverride({torch.bfloat16: 5e-2, torch.half: 1e-3})
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    @dtypesIfCUDA(torch.float, torch.double, torch.bfloat16, torch.half, torch.long)
    @dtypesIfCPU(torch.float, torch.double, torch.long)
    def test_signal_window_functions(self, device, dtype):

        def test(name, kwargs):
            torch_method = getattr(torch, name + '_window')
            if not dtype.is_floating_point:
                with self.assertRaisesRegex(RuntimeError, r'floating point'):
                    torch_method(3, dtype=dtype)
                return
            for size in [0, 1, 2, 5, 10, 50, 100, 1024, 2048]:
                for periodic in [True, False]:
                    res = torch_method(size, periodic=periodic, **kwargs, device=device, dtype=dtype)
                    # NB: scipy always returns a float64 result
                    ref = torch.from_numpy(signal.get_window((name, *(kwargs.values())), size, fftbins=periodic))
                    self.assertEqual(res, ref, exact_dtype=False)
            with self.assertRaisesRegex(RuntimeError, r'not implemented for sparse types'):
                torch_method(3, layout=torch.sparse_coo)
            self.assertTrue(torch_method(3, requires_grad=True).requires_grad)
            self.assertFalse(torch_method(3).requires_grad)

        for window in ['hann', 'hamming', 'bartlett', 'blackman']:
            test(window, kwargs={})

        for num_test in range(50):
            test('kaiser', kwargs={'beta': random.random() * 30})

instantiate_device_type_tests(TestSignal, globals())

if __name__ == '__main__':
    run_tests()