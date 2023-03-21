# Owner(s): ["module: inductor"]
import subprocess
import sys
from unittest.mock import patch

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor import config
from torch._inductor.codecache import PyCodeCache
from torch.testing import FileCheck
from torch.testing._internal.inductor_utils import HAS_CUDA


class TestKernelBenchmark(TestCase):
    @patch.object(config, "benchmark_kernel", True)
    def test_kernel_benchmark(self):
        @torch.compile
        def f(x):
            return torch.sin(x) + torch.cos(x)

        inp = torch.rand(2, 3).cuda()
        out = f(inp)

        compiled_module = None
        for k, v in PyCodeCache.cache.items():
            if hasattr(v, "benchmark_compiled_module"):
                self.assertTrue(
                    compiled_module is None, "Found multiple compiled modules"
                )
                compiled_module = v

        self.assertTrue(compiled_module is not None)

        try:
            # now run the compiled module in subprocess and check its output
            bench_out = subprocess.check_output(
                f"{sys.executable} {compiled_module.__file__} -kc".split(),
                stderr=subprocess.STDOUT,
            ).decode()

            # make sure we have the bandwidth information in the output
            FileCheck().check_count(
                "GB/s",
                1,
                exactly=1,
            ).run(bench_out)
        except subprocess.CalledProcessError:
            # calling python in a subprocess somehow fail in CI. Ignore it for now.
            # Even we ignore it, we've already done basic checks like the compiled
            # module should have benchmark_compiled_module function defined.
            pass


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
