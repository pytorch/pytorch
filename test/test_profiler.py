import collections
import gc
import unittest

import torch
from torch.testing._internal.common_utils import (
    TestCase, run_tests, TEST_WITH_ASAN, IS_WINDOWS)
from torch.autograd.profiler import profile

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
@unittest.skipIf(TEST_WITH_ASAN, "Cannot test with ASAN")
@unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestProfiler_cuda(TestCase):
    def test_mem_leak(self):
        """Checks that there's no memory leak when using profiler with CUDA
        """
        t = torch.rand(1, 1).cuda()
        p = psutil.Process()
        last_rss = collections.deque(maxlen=5)
        for outer_idx in range(10):
            with profile(use_cuda=True):
                for _ in range(1024):
                    t = torch.mm(t, t)

            gc.collect()
            torch.cuda.empty_cache()
            last_rss.append(p.memory_info().rss)

        max_diff = -1
        for idx in range(1, len(last_rss)):
            max_diff = max(max_diff, last_rss[idx] - last_rss[idx - 1])

        # with CUDA events leaking the increase in memory was ~7 MB,
        # using much smaller threshold but not zero to reduce flakiness
        self.assertTrue(max_diff < 100 * 1024)

if __name__ == '__main__':
    run_tests()
