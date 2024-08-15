# Owner(s): ["module: inductor"]
import os

import triton

from torch._inductor import config
from torch._inductor.test_case import run_tests
from torch.testing._internal.inductor_utils import HAS_CPU


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor

TRITON_HAS_CPU = "cpu" in triton.backends.backends


class TritonCpuTestMixin:
    @classmethod
    def setUpClass(cls):
        os.environ["TRITON_CPU_BACKEND"] = "1"

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("TRITON_CPU_BACKEND", None)


if HAS_CPU and TRITON_HAS_CPU:

    @config.patch(cpu_backend="triton")
    class SweepInputsCpuTritonTest(
        TritonCpuTestMixin, test_torchinductor.SweepInputsCpuTest
    ):
        pass

    @config.patch(cpu_backend="triton")
    class CpuTritonTests(TritonCpuTestMixin, test_torchinductor.CpuTests):
        pass


if __name__ == "__main__":
    if HAS_CPU and TRITON_HAS_CPU:
        run_tests(needs="filelock")
