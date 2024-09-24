# Owner(s): ["module: inductor"]
from torch._inductor import config
from torch._inductor.test_case import run_tests
from torch.testing._internal.inductor_utils import HAS_CPU
from torch.utils._triton import has_triton


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor

if has_triton():
    import triton

    TRITON_HAS_CPU = "cpu" in triton.backends.backends
else:
    TRITON_HAS_CPU = False


if HAS_CPU and TRITON_HAS_CPU:

    @config.patch(cpu_backend="triton")
    class SweepInputsCpuTritonTest(test_torchinductor.SweepInputsCpuTest):
        pass

    @config.patch(cpu_backend="triton")
    class CpuTritonTests(test_torchinductor.TestCase):
        common = test_torchinductor.check_model
        device = "cpu"

    test_torchinductor.copy_tests(
        test_torchinductor.CommonTemplate,
        CpuTritonTests,
        "cpu",
        xfail_prop="_expected_failure_triton_cpu",
    )


if __name__ == "__main__":
    if HAS_CPU and TRITON_HAS_CPU:
        run_tests(needs="filelock")
