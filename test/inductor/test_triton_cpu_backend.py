# Owner(s): ["module: inductor"]
import unittest

from torch._inductor import config
from torch._inductor.test_case import run_tests
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skip,
    skipOps,
)
from torch.testing._internal.inductor_utils import HAS_CPU, TRITON_HAS_CPU


try:
    from . import test_torchinductor, test_torchinductor_opinfo
except ImportError:
    import test_torchinductor
    import test_torchinductor_opinfo

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
    # ~50-150s
    "test_large_strided_reduction_cpu",
    "test_large_block_sizes_cpu",
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

    class TestInductorOpInfoTriton(test_torchinductor_opinfo.InductorOpInfoTemplate):
        # Start Triton CPU OpInfo coverage with index_add only. Other ops will
        # be enabled over time
        test_comprehensive = config.patch(
            {
                "cpu_backend": "triton",
                "test_configs.runtime_triton_dtype_assert": False,
                "test_configs.runtime_triton_shape_assert": False,
            }
        )(
            skipOps(
                {
                    skip(op.name, op.variant_test_name or "", device_type="cpu")
                    for op in test_torchinductor_opinfo.op_db[
                        test_torchinductor_opinfo.START : test_torchinductor_opinfo.END
                    ]
                    if op.full_name != "index_add"
                }
            )(test_torchinductor_opinfo.InductorOpInfoTemplate.test_comprehensive)
        )

    instantiate_device_type_tests(
        TestInductorOpInfoTriton,
        globals(),
        only_for="cpu",
    )


if __name__ == "__main__":
    if HAS_CPU and TRITON_HAS_CPU:
        run_tests(needs="filelock")
