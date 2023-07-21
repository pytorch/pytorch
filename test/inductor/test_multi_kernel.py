# Owner(s): ["module: inductor"]
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA
import torch

from torch._inductor import config

config.triton.multi_kernel = True
# config.triton.persistent_reductions = False
config.benchmark_kernel = True

class MultiKernelTest(TestCase):
    def test_softmax(self):
        x = torch.rand(2, 1024).cuda()
        ref = torch.softmax(x, -1)
        act = torch.compile(torch.softmax)(x, -1)
        self.assertTrue(torch.allclose(ref, act))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    if HAS_CUDA:
        run_tests()
