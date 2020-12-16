import torch

from torch.testing._internal.common_utils import (TestCase, run_tests)
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class TestTestRunner(TestCase):

    # Test that wrap_with_cuda_memory_check successfully detects leak
    def test_throw_unrecoverable_cuda_exception(self, device):
        x = torch.rand(10, device=device)
        # cause unrecoverable CUDA exception, recoverable on CPU
        y = x[torch.tensor([25])].cpu()

    def test_trivial_passing_case_cpu_cuda(self, device):
        x1 = torch.tensor([1., 0., 1., 0., 1., 0.], device=device)
        x2 = torch.tensor([1., 0., 1., 0., 1., 0.], device='cpu')
        self.assertEqual(x1, x2)


instantiate_device_type_tests(
    TestTestRunner,
    globals(),
    except_for=None
)

if __name__ == '__main__':
    run_tests()
