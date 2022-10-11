# Owner(s): ["module: signal"]

import torch
import unittest
import re

from torch.testing._internal.common_utils import (
    TestCase, run_tests
)
from torch.testing._internal.common_device_type import (
    ops, instantiate_device_type_tests, OpDTypes
)
from torch.testing._internal.common_methods_invocations import (
    precisionOverride, op_db
)
from torch.testing._internal.opinfo.core import OpInfo


class TestSignalWindows(TestCase):
    exact_dtype = False

    supported_windows = 'cosine|exponential|gaussian'

    def _test_window(self, device, dtype, op: OpInfo, **kwargs):
        if op.ref is None:
            raise unittest.SkipTest("No reference implementation")

        sample_inputs = op.sample_inputs(device, dtype, False, **kwargs)

        for sample_input in sample_inputs:
            window_size = sample_input.input
            window_name = re.search(self.supported_windows, op.name).group(0)

            ref_kwargs = {
                k: sample_input.kwargs[k] for k in sample_input.kwargs
                if k not in ('device', 'dtype', 'requires_grad', 'periodic')
            }

            expected = torch.from_numpy(
                op.ref((window_name, *(ref_kwargs.values())), window_size, fftbins=sample_input.kwargs['periodic'])
            )
            actual = op(window_size, **sample_input.kwargs)
            self.assertEqual(actual, expected, exact_dtype=self.exact_dtype)

        self.assertTrue(op(3, requires_grad=True).requires_grad)
        self.assertFalse(op(3).requires_grad)

    def _test_window_errors(self, device, op):
        error_inputs = op.error_inputs(device)

        for error_input in error_inputs:
            sample_input = error_input.sample_input
            with self.assertRaisesRegex(error_input.error_type, error_input.error_regex):
                op(sample_input.input, *sample_input.args, **sample_input.kwargs)

    @ops([op for op in op_db if 'windows' in op.name], dtypes=OpDTypes.none)
    def test_window_errors(self, device, op):
        self._test_window_errors(device, op)

    @precisionOverride({torch.bfloat16: 5e-2, torch.half: 1e-3})
    @ops([op for op in op_db if 'windows' in op.name],
         allowed_dtypes=(torch.float, torch.double))
    def test_windows(self, device, dtype, op):
        self._test_window(device, dtype, op)


instantiate_device_type_tests(TestSignalWindows, globals())

if __name__ == '__main__':
    run_tests()
