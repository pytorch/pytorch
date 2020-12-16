#!/usr/bin/env python

import torch

from torch.testing._internal.common_utils import (TestCase, run_tests)
from torch.testing._internal.common_device_type import instantiate_device_type_tests

# This test is added to ensure that test suite early terminates when 
# CUDA assert was thrown since all subsequence test will fail. 
# See: https://github.com/pytorch/pytorch/issues/49019
# This test file should be invoked from test_tes_runner.py
class TestThatContainsCUDAAssertFailure(TestCase):

    # Test that wrap_with_cuda_memory_check successfully detects leak
    def test_throw_unrecoverable_cuda_exception(self, device):
        x = torch.rand(10, device=device)
        # cause unrecoverable CUDA exception, recoverable on CPU
        y = x[torch.tensor([25])].cpu()

    def test_trivial_passing_test_case_on_cpu_cuda(self, device):
        x1 = torch.tensor([0., 1.], device=device)
        x2 = torch.tensor([0., 1.], device='cpu')
        self.assertEqual(x1, x2)


instantiate_device_type_tests(
    TestThatContainsCUDAAssertFailure,
    globals(),
    except_for=None
)

if __name__ == '__main__':
    run_tests()
