# Owner(s): ["module: inductor"]
from torch._inductor import config
from torch._inductor.test_case import run_tests
from torch.testing._internal.inductor_utils import HAS_CPU, TRITON_HAS_CPU


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor


if HAS_CPU and TRITON_HAS_CPU:

    @config.patch(cpu_backend="triton")
    class SweepInputsCpuTritonTest(test_torchinductor.SweepInputsCpuTest):
        pass


if __name__ == "__main__":
    if HAS_CPU and TRITON_HAS_CPU:
        run_tests(needs="filelock")
