# Owner(s): ["module: inductor"]
import sys
import unittest
from typing import NamedTuple

import torch._dynamo
from torch._inductor import config
from torch.testing._internal.common_utils import (
    IS_MACOS,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase as TorchTestCase,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA


try:
    try:
        from . import test_cpu_repro, test_mkldnn_pattern_matcher, test_torchinductor
    except ImportError:
        import test_cpu_repro
        import test_mkldnn_pattern_matcher
        import test_torchinductor
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


RUN_CPU = HAS_CPU and not torch.backends.mps.is_available() and not IS_MACOS
RUN_CUDA = HAS_CUDA and not TEST_WITH_ASAN and not TEST_WITH_ROCM


class CppWrapperTemplate:
    pass


class CudaWrapperTemplate:
    pass


class TestCppWrapper(TorchTestCase):
    device = "cpu"


class TestCudaWrapper(TorchTestCase):
    device = "cuda"


def make_test_case(name, device, tests, condition=True):
    test_name = f"{name}_{device}" if device else name

    @config.patch(cpp_wrapper=True, search_autotune_cache=False)
    def fn(self):
        tests.setUpClass()
        tests.setUp()
        try:
            func = getattr(tests, test_name)
            assert callable(func), "not a callable"
            code = test_torchinductor.run_and_get_cpp_code(func)
            self.assertEqual("load_inline" in code, True)
        finally:
            tests.tearDown()
            tests.tearDownClass()

    fn.__name__ = test_name
    if condition:
        setattr(
            CppWrapperTemplate if device != "cuda" else CudaWrapperTemplate,
            test_name,
            fn,
        )


if RUN_CPU:

    class BaseTest(NamedTuple):
        name: str
        device: str = "cpu"
        tests: TorchTestCase = test_torchinductor.CpuTests()
        condition: bool = True

    for item in [
        BaseTest("test_as_strided"),  # buffer reuse
        BaseTest("test_bitwise"),  # int32
        BaseTest("test_bmm1"),
        BaseTest("test_bmm2"),
        BaseTest("test_cat"),  # alias
        BaseTest("test_dtype_sympy_expr"),
        BaseTest("test_embedding_bag"),  # test default FallbackKernel
        BaseTest("test_index_put_deterministic_fallback"),
        BaseTest("test_int_div", "", test_cpu_repro.CPUReproTests()),
        BaseTest("test_linear1"),
        BaseTest("test_linear2"),
        BaseTest(
            "test_linear_binary",
            "",
            test_mkldnn_pattern_matcher.TestPaternMatcher(),
            torch._C.has_mkldnn and torch.ops.mkldnn._is_mkldnn_bf16_supported(),
        ),
        BaseTest("test_linear_packed", "", test_cpu_repro.CPUReproTests()),
        BaseTest("test_mm_views"),
        BaseTest("test_profiler_mark_wrapper_call"),
        BaseTest("test_reduction1"),  # Reduction
        BaseTest("test_relu"),  # multiple inputs
        BaseTest("test_repeat_interleave", "", test_cpu_repro.CPUReproTests()),
        BaseTest("test_scalar_input"),
        BaseTest("test_silu"),  # single input, single output
        BaseTest("test_sort"),
        BaseTest("test_sum_dtype"),  # float64
        BaseTest("test_sum_int"),  # bool, int64, int8, uint8
        BaseTest("test_transpose"),  # multiple outputs, buffer clear
    ]:
        make_test_case(item.name, item.device, item.tests, item.condition)

    test_torchinductor.copy_tests(CppWrapperTemplate, TestCppWrapper, "cpp_wrapper")

if RUN_CUDA:

    class BaseTest(NamedTuple):
        name: str
        device: str = "cuda"
        tests: TorchTestCase = test_torchinductor.CudaTests()

    # Maintain two separate test lists for cuda and cpp for now
    for item in [
        BaseTest("test_as_strided"),  # buffer reuse
        BaseTest("test_bitwise"),  # int32
        BaseTest("test_bmm1"),
        BaseTest("test_bmm2"),
        BaseTest("test_cat"),  # alias
        BaseTest("test_convolution1"),
        BaseTest("test_conv_backward"),
        BaseTest("test_embedding_bag"),  # test default FallbackKernel
        BaseTest("test_index_put_deterministic_fallback"),
        BaseTest("test_linear1"),
        BaseTest("test_linear2"),
        BaseTest("test_mm_views"),
        BaseTest("test_multi_device"),
        BaseTest("test_profiler_mark_wrapper_call"),
        BaseTest("test_reduction1"),  # Reduction
        BaseTest("test_relu"),  # multiple inputs
        BaseTest("test_scalar_input"),
        BaseTest("test_sort"),
        BaseTest("test_silu"),  # single input, single output
        BaseTest("test_sum_dtype"),  # float64
        BaseTest("test_sum_int"),  # bool, int64, int8, uint8
        BaseTest("test_transpose"),  # multiple outputs, buffer clear
    ]:
        make_test_case(item.name, item.device, item.tests)

    test_torchinductor.copy_tests(CudaWrapperTemplate, TestCudaWrapper, "cuda_wrapper")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if RUN_CPU or RUN_CUDA:
        run_tests(needs="filelock")
