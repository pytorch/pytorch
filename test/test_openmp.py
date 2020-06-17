import collections
import unittest

import numpy as np
import torch
from torch.testing._internal.common_utils import (
    TestCase, run_tests, TEST_WITH_ASAN)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

device = torch.device('cpu')


class Network(torch.nn.Module):
    maxp1 = torch.nn.MaxPool2d(1, 1)

    def forward(self, x):
        return self.maxp1(x)


@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
@unittest.skipIf(TEST_WITH_ASAN, "Cannot test with ASAN")
class TestOpenMP_ParallelFor(TestCase):
    batch = 20
    channels = 1
    side_dim = 80
    x = torch.randn([batch, channels, side_dim, side_dim], device=device)
    model = Network()
    ncores = min(5, psutil.cpu_count(logical=False))

    def func(self, runs):
        p = psutil.Process()
        # warm up for 5 runs, then things should be stable for the last 5
        last_rss = collections.deque(maxlen=5)
        for n in range(10):
            for i in range(runs):
                self.model(self.x)
            last_rss.append(p.memory_info().rss)
        return last_rss

    def func_rss(self, runs):
        last_rss = self.func(runs)
        # Do a least-mean-squares fit of last_rss to a line
        poly = np.polynomial.Polynomial.fit(
            range(len(last_rss)), np.array(last_rss), 1)
        coefs = poly.convert().coef
        # The coefs are (b, m) for the line y = m * x + b that fits the data.
        # If m == 0 it will not be present. Assert it is missing or < 1000.
        self.assertTrue(len(coefs) < 2 or coefs[1] < 1000,
                        msg='memory did not stabilize, {}'.format(str(list(last_rss))))

    def test_one_thread(self):
        """Make sure there is no memory leak with one thread: issue gh-32284
        """
        torch.set_num_threads(1)
        self.func_rss(300)

    def test_n_threads(self):
        """Make sure there is no memory leak with many threads
        """
        torch.set_num_threads(self.ncores)
        self.func_rss(300)

if __name__ == '__main__':
    run_tests()
