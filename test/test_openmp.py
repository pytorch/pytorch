# Owner(s): ["module: unknown"]

import collections
import os
import subprocess
import sys
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    import psutil

    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False
    psutil = None


device = torch.device("cpu")


class Network(torch.nn.Module):
    maxp1 = torch.nn.MaxPool2d(1, 1)

    def forward(self, x):
        return self.maxp1(x)


@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
class TestOpenMP_ParallelFor(TestCase):
    batch = 20
    channels = 1
    side_dim = 80
    x = torch.randn([batch, channels, side_dim, side_dim], device=device)
    model = Network()

    def func(self, runs):
        p = psutil.Process()
        # warm up for 5 runs, then things should be stable for the last 5
        last_rss = collections.deque(maxlen=5)
        for _ in range(10):
            for _ in range(runs):
                self.model(self.x)
            last_rss.append(p.memory_info().rss)
        return last_rss

    def func_rss(self, runs):
        last_rss = list(self.func(runs))
        # Check that the sequence is not strictly increasing
        is_increasing = True
        for idx in range(len(last_rss)):
            if idx == 0:
                continue
            is_increasing = is_increasing and (last_rss[idx] > last_rss[idx - 1])
        self.assertTrue(
            not is_increasing,
            msg=lambda msg: f"{msg}\nmemory usage is increasing, {str(last_rss)}",
        )

    def test_one_thread(self):
        """Make sure there is no memory leak with one thread: issue gh-32284"""
        torch.set_num_threads(1)
        self.func_rss(300)

    def test_n_threads(self):
        """Make sure there is no memory leak with many threads"""
        ncores = min(5, psutil.cpu_count(logical=False))
        torch.set_num_threads(ncores)
        self.func_rss(300)


@unittest.skipIf(sys.platform != "win32", "OMP_WAIT_POLICY default is Windows-only")
class TestOpenMPWaitPolicyDefault(TestCase):
    """`import torch` should default OMP_WAIT_POLICY to passive on Windows so
    idle OpenMP workers sleep instead of busy-spinning (gh-177906), while never
    overriding a preference the user set themselves."""

    _PROBE = "import os, torch; print('OMPWP=' + os.environ.get('OMP_WAIT_POLICY', ''))"

    def _omp_wait_policy_after_import(self, env_overrides):
        env = os.environ.copy()
        env.pop("OMP_WAIT_POLICY", None)
        env.pop("KMP_BLOCKTIME", None)
        env.update(env_overrides)
        out = subprocess.check_output(
            [sys.executable, "-c", self._PROBE], env=env, text=True
        )
        marker = next(ln for ln in out.splitlines() if ln.startswith("OMPWP="))
        return marker[len("OMPWP=") :]

    def test_default_is_passive(self):
        self.assertEqual(self._omp_wait_policy_after_import({}), "PASSIVE")

    def test_user_wait_policy_is_respected(self):
        self.assertEqual(
            self._omp_wait_policy_after_import({"OMP_WAIT_POLICY": "active"}), "active"
        )

    def test_user_blocktime_disables_default(self):
        # A user who tunes KMP_BLOCKTIME has opted into managing the wait
        # behavior, so we must not also force a wait policy on top of it.
        self.assertEqual(self._omp_wait_policy_after_import({"KMP_BLOCKTIME": "1"}), "")


if __name__ == "__main__":
    run_tests()
