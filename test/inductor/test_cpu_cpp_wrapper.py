# Owner(s): ["oncall: cpu inductor"]
import sys
import unittest
from typing import NamedTuple

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.common_device_type import (
    get_desired_device_type_test_bases,
)
from torch.testing._internal.common_utils import (
    IS_MACOS,
    IS_WINDOWS,
    slowTest,
    TEST_MKL,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import HAS_CPU


try:
    try:
        from . import (
            test_cpu_repro,
            test_cpu_select_algorithm,
            test_mkldnn_pattern_matcher,
            test_torchinductor,
            test_torchinductor_dynamic_shapes,
        )
    except ImportError:
        import test_cpu_repro  # @manual=fbcode//caffe2/test/inductor:test_cpu_repro-library
        import test_cpu_select_algorithm  # @manual=fbcode//caffe2/test/inductor:cpu_select_algorithm_cpu-library
        import test_mkldnn_pattern_matcher  # @manual
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
        import test_torchinductor_dynamic_shapes  # @manual=fbcode//caffe2/test/inductor:test_inductor-library_dynamic_shapes
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


_desired_test_bases = get_desired_device_type_test_bases()
RUN_CPU = (
    HAS_CPU
    and any(getattr(x, "device_type", "") == "cpu" for x in _desired_test_bases)
    and not IS_MACOS
)


class CppWrapperTemplate:
    pass


class TestCppWrapper(InductorTestCase):
    device = "cpu"


class DynamicShapesCppWrapperCpuTests(InductorTestCase):
    device = "cpu"


test_failures_cpp_wrapper = {
    # conv2d will fallback for dynamic shapes; the fallback path is not yet supported
    "test_conv2d_unary_cpu_dynamic_shapes": test_torchinductor.TestFailure(
        ("cpp_wrapper",), is_skip=True
    ),
    "test_conv2d_binary_inplace_fusion_failed_cpu_dynamic_shapes": test_torchinductor.TestFailure(
        ("cpp_wrapper",), is_skip=True
    ),
    "test_conv2d_binary_inplace_fusion_pass_cpu_dynamic_shapes": test_torchinductor.TestFailure(
        ("cpp_wrapper",), is_skip=True
    ),
    # aten._native_multi_head_attention.default is not yet supported for dynamic shapes
    "test_multihead_attention_cpu_dynamic_shapes": test_torchinductor.TestFailure(
        ("cpp_wrapper",), is_skip=True
    ),
}
if TEST_WITH_ROCM:
    test_failures_cpp_wrapper.update(
        {
            "test_linear_packed": test_torchinductor.TestFailure(
                ("cpp_wrapper"), is_skip=True
            ),
            "test_linear_packed_dynamic_shapes": test_torchinductor.TestFailure(
                ("cpp_wrapper"), is_skip=True
            ),
        }
    )


def make_test_case(
    name,
    device,
    tests,
    condition=True,
    slow=False,
    func_inputs=None,
    code_string_count=None,
    test_build_separate=False,
):
    test_name = f"{name}_{device}" if device else name
    if code_string_count is None:
        code_string_count = {}

    func = getattr(tests, test_name)
    assert callable(func), "not a callable"
    func = slowTest(func) if slow else func
    new_test_name = f"{test_name}_separate" if test_build_separate else test_name

    @config.patch(
        cpp_wrapper=True,
        cpp_wrapper_build_separate=test_build_separate,
    )
    def fn(self):
        tests.setUpClass()
        tests.setUp()
        try:
            with torch._C._PreserveDispatchKeyGuard():
                torch._C._dispatch_tls_set_dispatch_key_included(
                    torch._C.DispatchKey.Dense, True
                )

                _, code = test_torchinductor.run_and_get_cpp_code(
                    func, *func_inputs if func_inputs else []
                )
                # If a test generates no code, skip the remaining checks.  This can
                # happen for tests validating build-dependent features (e.g. datatypes
                # that are available on some platforms and not others).
                if code:
                    if test_build_separate:
                        self.assertIn("kernel_src", code)
                    self.assertIn("CppWrapperCodeCache", code)
                    self.assertTrue(
                        all(
                            code.count(string) == code_string_count[string]
                            for string in code_string_count
                        )
                    )
        finally:
            tests.tearDown()
            tests.tearDownClass()

    fn.__name__ = new_test_name
    import copy

    fn.__dict__ = copy.deepcopy(func.__dict__)
    if condition:
        setattr(
            CppWrapperTemplate,
            new_test_name,
            fn,
        )


if RUN_CPU:

    class BaseTest(NamedTuple):
        name: str
        device: str = "cpu"
        tests: InductorTestCase = test_torchinductor.CpuTests()
        condition: bool = True
        slow: bool = False
        func_inputs: list = None
        code_string_count: dict = {}
        test_build_separate: bool = False

    for item in [
        BaseTest("test_add_complex"),
        BaseTest("test_add_complex", test_build_separate=True),
        BaseTest("test_add_complex4"),
        BaseTest("test_add_complex4", test_build_separate=True),
        BaseTest("test_as_strided"),  # buffer reuse
        BaseTest("test_bernoulli1_combo_kernels_False"),
        BaseTest("test_bernoulli1_combo_kernels_True"),
        BaseTest("test_bitwise"),  # int32
        BaseTest("test_bmm1"),
        BaseTest("test_bmm1", test_build_separate=True),
        BaseTest("test_bmm2"),
        BaseTest("test_cat"),  # alias
        BaseTest(
            "test_conv2d_unary",
            "cpu",
            test_mkldnn_pattern_matcher.TestPatternMatcherGenericCPU(),
            condition=torch.backends.mkldnn.is_available(),
            slow=True,
        ),
        BaseTest("test_conv_transpose2d_packed", "cpu", test_cpu_repro.CPUReproTests()),
        BaseTest("test_cumsum"),
        BaseTest("test_custom_op_1"),
        BaseTest("test_custom_op_2"),
        BaseTest("test_custom_op_3"),
        BaseTest("test_dtype_sympy_expr"),
        BaseTest("test_embedding_bag"),  # test default FallbackKernel
        BaseTest("test_index_put1"),
        BaseTest("test_index_put_deterministic_fallback"),
        BaseTest("test_adding_tensor_offsets"),
        BaseTest("test_inductor_layout_optimization_input_mutations"),
        BaseTest("test_int_div", "", test_cpu_repro.CPUReproTests()),
        BaseTest("test_int8_weight_only_quant"),
        BaseTest("test_linear1"),
        BaseTest("test_linear2"),
        *[
            BaseTest(func, "", test_cpu_select_algorithm.TestSelectAlgorithmCPU())
            for func in dir(test_cpu_select_algorithm.TestSelectAlgorithmCPU())
            if func.startswith(
                (
                    # skip for now since it's flaky:
                    # https://github.com/pytorch/pytorch/actions/runs/19916391966/job/57096613509?pr=169151
                    # "test_linear_with_pointwise",
                    "test_grouped_linear",
                )
            )
        ],
        BaseTest("test_polar"),
        BaseTest(
            "test_linear_binary",
            "",
            test_mkldnn_pattern_matcher.TestPatternMatcher(),
            torch.backends.mkldnn.is_available()
            and torch.ops.mkldnn._is_mkldnn_bf16_supported(),
        ),
        BaseTest(
            "test_linear_packed",
            "",
            test_cpu_repro.CPUReproTests(),
            torch.backends.mkldnn.is_available()
            and (
                torch.ops.mkldnn._is_mkldnn_bf16_supported()
                or torch.ops.mkldnn._is_mkldnn_fp16_supported()
            ),
        ),
        *[
            BaseTest(
                func,
                "",
                test_cpu_repro.CPUReproTests(),
                condition=torch.backends.mkldnn.is_available() and not IS_WINDOWS,
            )
            for func in dir(test_cpu_repro.CPUReproTests())
            if func.startswith("test_lstm_packed_change_input_sizes")
        ],
        BaseTest("test_max_pool2d6_dilation_1"),
        BaseTest("test_max_pool2d6_dilation_2"),
        BaseTest(
            "test_mkl_linear", "", test_cpu_repro.CPUReproTests(), condition=TEST_MKL
        ),
        BaseTest("test_mm_views"),
        BaseTest("test_multihead_attention", "cpu", test_cpu_repro.CPUReproTests()),
        BaseTest(
            "test_multi_threading",
            condition=not IS_WINDOWS,
            # Two threads compile, so we expect the output code to be printed twice.
            code_string_count={"py::gil_scoped_release_simple release;": 2},
        ),
        BaseTest("test_profiler_mark_wrapper_call"),
        BaseTest("test_randint"),
        BaseTest("test_randn_with_dtype_and_device"),
        BaseTest("test_reduction1"),  # Reduction
        BaseTest("test_relu"),  # multiple inputs
        BaseTest("test_repeat_interleave", "", test_cpu_repro.CPUReproTests()),
        BaseTest("test_scalar_input"),
        BaseTest("test_scalar_output"),
        BaseTest("test_scaled_dot_product_attention"),
        BaseTest("test_scatter1"),
        BaseTest("test_scatter2"),
        BaseTest("test_scatter3"),
        BaseTest("test_scatter4"),
        BaseTest("test_scatter5"),
        BaseTest("test_scatter6"),
        BaseTest("test_scatter_reduce1"),
        BaseTest("test_scatter_reduce2"),
        BaseTest("test_scatter_reduce3"),
        BaseTest("test_silu"),  # single input, single output
        BaseTest("test_sort"),
        BaseTest("test_sum_dtype"),  # float64
        BaseTest("test_sum_int"),  # bool, int64, int8, uint8
        BaseTest("test_tensor2"),  # constant input
        BaseTest(
            "test_transpose", code_string_count={".reset();": 2}
        ),  # multiple outputs, buffer clear
        BaseTest("test_view_as_complex"),
        BaseTest("test_view_as_real"),
    ]:
        make_test_case(
            item.name,
            item.device,
            item.tests,
            item.condition,
            item.slow,
            item.func_inputs,
            item.code_string_count,
            item.test_build_separate,
        )

    test_torchinductor.copy_tests(
        CppWrapperTemplate,
        TestCppWrapper,
        "cpp_wrapper",
        test_failures_cpp_wrapper,
    )

    DynamicShapesCppWrapperTemplate = (
        test_torchinductor_dynamic_shapes.make_dynamic_cls(CppWrapperTemplate)
    )

    test_torchinductor.copy_tests(
        DynamicShapesCppWrapperTemplate,
        DynamicShapesCppWrapperCpuTests,
        "cpp_wrapper",
        test_failures_cpp_wrapper,
        xfail_prop="_expected_failure_dynamic_wrapper",
    )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_CPU:
        run_tests(needs="filelock")
