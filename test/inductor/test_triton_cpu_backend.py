# Owner(s): ["module: inductor"]
import unittest

from torch._inductor import config
from torch._inductor.test_case import run_tests
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
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

    # TODO: support generating inductor backend subclasses in instantiate_device_type_tests
    def make_inductor_opinfo_triton_cpu_cls():
        ops_subset = [
            next(
                op
                for op in test_torchinductor_opinfo.op_db
                if op.full_name == "index_add"
            )
        ]
        # Clone the base opinfo class and use that in `instantiate_device_type_tests`
        # in order to preserve DecorateInfo references to TestTorchInductorOpInfo
        TestTorchInductorOpInfo = test_torchinductor_opinfo.make_inductor_opinfo_cls(
            test_torchinductor_opinfo._ops(ops_subset),
            skipOps(test_torchinductor_opinfo.test_skips_or_fails),
        )
        opinfo_scope = {
            TestTorchInductorOpInfo.__name__: TestTorchInductorOpInfo,
        }

        instantiate_device_type_tests(
            TestTorchInductorOpInfo, opinfo_scope, only_for="cpu"
        )

        cpu_cls_name = f"{TestTorchInductorOpInfo.__name__}CPU"
        if cpu_cls_name not in opinfo_scope:
            raise AssertionError(f"Expected {cpu_cls_name} in OpInfo test scope")
        generated_cpu_cls = opinfo_scope[cpu_cls_name]

        return config.patch({"cpu_backend": "triton"})(
            type("TestInductorOpInfoTritonCPU", (generated_cpu_cls,), {})
        )

    TestInductorOpInfoTritonCPU = make_inductor_opinfo_triton_cpu_cls()


if __name__ == "__main__":
    if HAS_CPU and TRITON_HAS_CPU:
        run_tests(needs="filelock")
