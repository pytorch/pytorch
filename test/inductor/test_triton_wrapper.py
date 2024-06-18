# Owner(s): ["module: inductor"]

import subprocess
import sys

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._inductor.codecache import PyCodeCache
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class TestTritonWrapper(TestCase):
    def get_compiled_module(self):
        compiled_module = None
        for v in PyCodeCache.cache.values():
            if hasattr(v, "benchmark_compiled_module"):
                self.assertTrue(
                    compiled_module is None, "Found multiple compiled modules"
                )
                compiled_module = v

        self.assertTrue(compiled_module is not None)
        return compiled_module

    def test_wrapper_using_gpu_seed(self):
        """
        Make sure the subprocess.check_output does not throw.
        """

        @torch.compile
        def f(x, y):
            # dropout will result in usage of cuda_seed
            z = torch.nn.functional.dropout(x, 0.5)
            return z + y

        N = 10
        x = torch.rand(N).to(device=GPU_TYPE)
        y = torch.rand(N).to(device=GPU_TYPE)
        out = f(x, y)
        compiled_module = self.get_compiled_module()

        # now run the compiled module in subprocess and check its output
        bench_out = subprocess.check_output(
            f"{sys.executable} {compiled_module.__file__}".split(),
            stderr=subprocess.STDOUT,
        ).decode()

        self.assertTrue(len(bench_out) > 0)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
