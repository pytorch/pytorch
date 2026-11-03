# Owner(s): ["module: inductor"]
import unittest

from torch._inductor import config
from torch._inductor.test_case import run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests
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
    # TODO: re-enable when the stable triton CPU backend can compile
    # inductor CPU code. The backend is newly present via the stable
    # channel, but inductor emits constructs it cannot compile yet:
    # missing libdevice.* lowering and no CPUDriver.tensor_descriptor.
    # Internal-only: stable is an old (3.5-era) frontend with a retrofitted
    # 3.8-based CPU backend; drop this once stable is upgraded to 3.8.
    _CPU_TRITON_SKIP_REASON = (
        "triton CPU backend in stable cannot yet compile inductor CPU code "
        "(missing libdevice lowering and tensor descriptor support)"
    )

    # SweepInputsCpuTest is only defined when CPU tests are enabled
    # (RUN_CPU); otherwise there is nothing to subclass.
    if hasattr(test_torchinductor, "SweepInputsCpuTest"):

        @unittest.skip(_CPU_TRITON_SKIP_REASON)
        @config.patch(
            {
                "cpu_backend": "triton",
                "test_configs.runtime_triton_dtype_assert": False,
                "test_configs.runtime_triton_shape_assert": False,
            }
        )
        class SweepInputsCpuTritonTest(test_torchinductor.SweepInputsCpuTest):
            pass

    @unittest.skip(_CPU_TRITON_SKIP_REASON)
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
        OPINFO_INCLUDE_OPS = {"index_add"}
        # TODO: expand op support over time. For now add as we see fit
        # to avoid increasing CI time too much. CPU OpInfo expectation tables
        # describe the C++ backend, so audit each addition for Triton CPU.
        ops_subset = [
            op
            for op in test_torchinductor_opinfo.op_db
            if op.full_name in OPINFO_INCLUDE_OPS
        ]
        included_ops = {op.full_name for op in ops_subset}
        if included_ops != OPINFO_INCLUDE_OPS:
            missing_ops = sorted(OPINFO_INCLUDE_OPS - included_ops)
            raise AssertionError(f"Missing OpInfo entries: {missing_ops}")

        # Generate a fresh source class so DecorateInfo matches TestInductorOpInfo.
        TestInductorOpInfo = test_torchinductor_opinfo.make_inductor_opinfo_cls(
            ops_subset
        )
        opinfo_scope = {
            TestInductorOpInfo.__name__: TestInductorOpInfo,
        }

        instantiate_device_type_tests(TestInductorOpInfo, opinfo_scope, only_for="cpu")

        generated_classes = [
            cls
            for name, cls in opinfo_scope.items()
            if name != TestInductorOpInfo.__name__
        ]
        if len(generated_classes) != 1:
            raise AssertionError(
                f"Expected exactly one generated OpInfo class, got {len(generated_classes)}"
            )
        generated_cpu_cls = generated_classes[0]

        @unittest.skip(_CPU_TRITON_SKIP_REASON)
        @config.patch({"cpu_backend": "triton"})
        class TestInductorOpInfoTritonCPU(generated_cpu_cls):
            pass

        TestInductorOpInfoTritonCPU.__qualname__ = "TestInductorOpInfoTritonCPU"
        return TestInductorOpInfoTritonCPU

    TestInductorOpInfoTritonCPU = make_inductor_opinfo_triton_cpu_cls()


if __name__ == "__main__":
    if HAS_CPU and TRITON_HAS_CPU:
        run_tests(needs="filelock")
