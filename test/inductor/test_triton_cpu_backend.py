# Owner(s): ["module: inductor"]
import unittest

from torch._inductor import config
from torch._inductor.test_case import run_tests
from torch.testing._internal.inductor_utils import HAS_CPU, TRITON_HAS_CPU


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor

TRITON_CPU_SLOW_TESTS = (
    # ~1000s
    "test_sort_stable_cpu",
    # ~300-400s
    "test_sort_bool_cpu",
    "test_sort_transpose_cpu",
    # ~100-300s
    "test_avg_pool3d_backward2_cpu",
    "test_pattern_matcher_multi_user_cpu",
    "test_split_cumsum_cpu",
)

if HAS_CPU and TRITON_HAS_CPU:

    @config.patch(
        {
            "cpu_backend": "triton",
            "test_configs.runtime_triton_dtype_assert": False,
            "test_configs.runtime_triton_shape_assert": False,
        }
    )
    class SweepInputsCpuTritonTest(test_torchinductor.SweepInputsCpuTest):
        pass

    @config.patch(
        {
            "cpu_backend": "triton",
            "test_configs.runtime_triton_dtype_assert": False,
            "test_configs.runtime_triton_shape_assert": False,
        }
    )
    class CpuTritonTests(test_torchinductor.TestCase):
        common = test_torchinductor.check_model
        device = "cpu"

    test_torchinductor.copy_tests(
        test_torchinductor.CommonTemplate,
        CpuTritonTests,
        "cpu",
        xfail_prop="_expected_failure_triton_cpu",
    )

    for name in TRITON_CPU_SLOW_TESTS:
        setattr(
            CpuTritonTests,
            name,
            unittest.skip("Triton CPU: slow test")(getattr(CpuTritonTests, name)),
        )


if __name__ == "__main__":
    if HAS_CPU and TRITON_HAS_CPU:
        run_tests(needs="filelock")
