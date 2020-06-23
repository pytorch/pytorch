import collections
import unittest

import torch
from torch.testing._internal.common_utils import (
    TestCase, run_tests, TEST_WITH_ASAN)
from torch.autograd.profiler import profile


@unittest.skipIf(TEST_WITH_ASAN, "Cannot test with ASAN")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestProfiler_cuda(TestCase):
    def test_cuda_mem_leak(self):
        """Checks that there's no CUDA memory leak when using the profiler
        """
        device = torch.device('cuda:0')
        t = torch.rand(1, 1).cuda(device=device)
        stats = torch.cuda.memory_stats(device)
        print("stats before: ", stats)
        with profile(use_cuda=True):
            for _ in range(1024):
                t = torch.mm(t, t)
        stats = torch.cuda.memory_stats(device)
        print("\nstats after: ", stats)


if __name__ == '__main__':
    run_tests()
