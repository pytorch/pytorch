# Owner(s): ["module: inductor"]
import itertools
import sys
import unittest
from typing import NamedTuple

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.common_utils import slowTest
from torch.testing._internal.inductor_utils import GPU_TYPE, RUN_GPU


try:
    try:
        from . import (
            test_combo_kernels,
            test_foreach,
            test_pattern_matcher,
            test_select_algorithm,
            test_torchinductor,
            test_torchinductor_dynamic_shapes,
        )
    except ImportError:
        import test_combo_kernels  # @manual=fbcode//caffe2/test/inductor:combo_kernels-library

        import test_foreach  # @manual=fbcode//caffe2/test/inductor:foreach-library
        import test_pattern_matcher  # @manual=fbcode//caffe2/test/inductor:pattern_matcher-library
        import test_select_algorithm  # @manual=fbcode//caffe2/test/inductor:select_algorithm-library
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
        import test_torchinductor_dynamic_shapes  # @manual=fbcode//caffe2/test/inductor:test_inductor-library_dynamic_shapes
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


class GpuWrapperTemplate:
    pass


class TestGpuWrapper(InductorTestCase):
    device = GPU_TYPE

    def test_aoti_debug_printer_works_on_constants(self):
        batch_size = 32
        seq_length = 50
        hidden_size = 768

        def test_fn():
            inp = torch.randn(batch_size, seq_length, hidden_size, device=self.device)
            weight = torch.randn(hidden_size, hidden_size, device=self.device)
            matmul_output = inp @ weight
            torch.nn.LayerNorm(hidden_size, device=self.device)(matmul_output)
            return True

        comp = torch.compile(
            options={
                "cpp_wrapper": True,
                "aot_inductor.debug_intermediate_value_printer": "2",
            }
        )(test_fn)
        comp()


class DynamicShapesGpuWrapperGpuTests(InductorTestCase):
    device = GPU_TYPE

    def test_annotation_training(self):
        batch_size = 32
        seq_length = 50
        hidden_size = 768

        def create_test_fn():
            def test_fn():
                inp = torch.randn(
                    batch_size, seq_length, hidden_size, device=self.device
                )
                weight = torch.randn(hidden_size, hidden_size, device=self.device)
                matmul_output = inp @ weight
                torch.nn.LayerNorm(hidden_size, device=self.device)(matmul_output)
                return True

            return test_fn

        fn = torch.compile(options={"annotate_training": True, "cpp_wrapper": True})(
            create_test_fn()
        )
        fn()


test_failures_gpu_wrapper = {
    "test_mm_plus_mm2_dynamic_shapes": test_torchinductor.TestFailure(
        ("gpu_wrapper",), is_skip=True
    ),
    "test_randint_xpu": test_torchinductor.TestFailure(("gpu_wrapper",), is_skip=False),
    "test_randint_xpu_dynamic_shapes": test_torchinductor.TestFailure(
        ("gpu_wrapper",), is_skip=False
    ),
    # ATen ops: scaled_dot_product_efficient_attention not implemented on XPU.
    "test_scaled_dot_product_efficient_attention_xpu": test_torchinductor.TestFailure(
        ("gpu_wrapper",), is_skip=False
    ),
    "test_scaled_dot_product_efficient_attention_xpu_dynamic_shapes": test_torchinductor.TestFailure(
        ("gpu_wrapper",), is_skip=False
    ),
}


def make_test_case(
    name,
    device,
    tests,
    condition=True,
    slow=False,
    func_inputs=None,
    code_string_count=None,
    check_code=True,
):
    test_name = f"{name}_{device}" if device else name
    if code_string_count is None:
        code_string_count = {}

    func = getattr(tests, test_name)
    assert callable(func), "not a callable"
    func = slowTest(func) if slow else func

    @config.patch(cpp_wrapper=True)
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
                if check_code:
                    self.assertEqual("CppWrapperCodeCache" in code, True)
                    self.assertTrue(
                        all(
                            code.count(string) == code_string_count[string]
                            for string in code_string_count
                        )
                    )
        finally:
            tests.tearDown()
            tests.tearDownClass()

    fn.__name__ = test_name
    import copy

    fn.__dict__ = copy.deepcopy(func.__dict__)
    if condition:
        setattr(
            GpuWrapperTemplate,
            test_name,
            fn,
        )


if RUN_GPU:

    class BaseTest(NamedTuple):
        name: str
        device: str = GPU_TYPE
        tests: InductorTestCase = test_torchinductor.GPUTests()
        check_code: bool = True

    # XPU Not implemented yet
    XPU_BASE_TEST_SKIP = [
        "test_foreach_cpp_wrapper",
        "test_enable_dynamic_shapes_cpp_wrapper",
        "test_dynamic_shapes_persistent_reduction_mixed_x_dim",
        "test_cat_slice_cat",
        "test_fft_real_input",
        "test_fft_real_input_real_output",
    ]

    # Maintain two separate test lists for cuda and cpp for now
    for item in [
        BaseTest("test_add_complex"),
        BaseTest("test_add_complex4"),
        BaseTest("test_as_strided"),  # buffer reuse
        BaseTest("test_batch_norm_2d_2"),
        BaseTest("test_bernoulli1"),
        BaseTest("test_bitwise"),  # int32
        BaseTest("test_bmm1"),
        BaseTest("test_bmm2"),
        BaseTest("test_buffer_use_after_remove"),
        BaseTest("test_cat"),  # alias
        BaseTest("test_convolution1"),
        BaseTest("test_conv_backward"),
        BaseTest("test_custom_op_1"),
        BaseTest("test_custom_op_2"),
        BaseTest("test_custom_op_3"),
        BaseTest("test_embedding_bag"),  # test default FallbackKernel
        BaseTest("test_index_put_deterministic_fallback"),
        BaseTest("test_adding_tensor_offsets"),
        BaseTest("test_index_tensor"),
        BaseTest("test_inductor_layout_optimization_input_mutations"),
        BaseTest("test_insignificant_strides"),
        BaseTest("test_layer_norm"),
        BaseTest("test_linear1"),
        BaseTest("test_linear2"),
        BaseTest("test_mm_views"),
        BaseTest("test_multi_device"),
        BaseTest("test_multi_threading"),
        BaseTest("test_pow3"),
        BaseTest("test_profiler_mark_wrapper_call"),
        BaseTest("test_randint"),
        BaseTest("test_reduction1"),  # Reduction
        BaseTest("test_relu"),  # multiple inputs
        BaseTest("test_repeat_interleave_2"),
        BaseTest("test_roi_align"),
        BaseTest("test_scalar_input"),
        BaseTest("test_scaled_dot_product_attention"),
        BaseTest("test_scaled_dot_product_efficient_attention"),
        BaseTest("test_sort"),
        BaseTest("test_silu"),  # single input, single output
        BaseTest("test_sum_dtype"),  # float64
        BaseTest("test_sum_int"),  # bool, int64, int8, uint8
        BaseTest("test_transpose"),  # multiple outputs, buffer clear
        *[
            BaseTest(f"test_unspec_inputs_{str(dtype)[6:]}")
            for dtype in test_torchinductor.test_dtypes
        ],
        BaseTest("test_consecutive_split_cumprod"),
        BaseTest("test_pointwise_hermite_polynomial_he"),
        BaseTest("test_pointwise_hermite_polynomial_h"),
        BaseTest(
            "test_foreach_cpp_wrapper",
            tests=test_foreach.ForeachTests(),
        ),  # test foreach
        BaseTest(
            "test_enable_dynamic_shapes_cpp_wrapper",
            tests=test_foreach.ForeachTests(),
        ),
        BaseTest(
            "test_dynamic_shapes_persistent_reduction_mixed_x_dim",
            tests=test_combo_kernels.ComboKernelDynamicShapesTests(),
        ),
        BaseTest(
            "test_cat_slice_cat",
            tests=test_pattern_matcher.TestPatternMatcher(),
        ),
        # TODO: Re-enable this test after fixing cuda wrapper for conv Triton templates with dynamic shapes.
        # This test is unstable: it succeeds when an ATEN kernel is used, and fails when a Triton kernel is used.
        # Currently it passes on CI (an ATEN kernel is chosen) and fails locally (a Triton kernel is chosen).
        # Ideally, it should succeed for whatever kernels.
        # BaseTest(
        #     "test_convolution1",
        #     device=None,
        #     tests=test_select_algorithm.TestSelectAlgorithm(),
        # ),
        BaseTest(
            "test_mm_plus_mm2",
            device=None,
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
        BaseTest(
            "test_mm_plus_mm3",
            device=None,
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
        BaseTest("test_fft_real_input"),
        BaseTest("test_fft_real_input_real_output"),
        *[
            # some dtypes may raise exception and be skipped in test_dtypeview, so set check_code to False here
            BaseTest(
                f"test_dtypeview_{str(dtype_x)[6:]}_{str(dtype_y)[6:]}",
                check_code=False,
            )
            for dtype_x, dtype_y in itertools.product(
                test_torchinductor.test_dtypes, test_torchinductor.test_dtypes
            )
        ],
        BaseTest("test_dtypeview_fusion"),
        # skip if not enough SMs
        BaseTest(
            "test_addmm",
            device=None,
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
        # skip if not enough SMs
        BaseTest(
            "test_linear_relu",
            device=None,
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
    ]:
        if item.device == "xpu" and item.name in XPU_BASE_TEST_SKIP:
            continue
        make_test_case(item.name, item.device, item.tests, check_code=item.check_code)

    from torch._inductor.utils import is_big_gpu

    if GPU_TYPE == "cuda" and is_big_gpu():
        skip_list = ["test_addmm", "test_linear_relu"]
        # need to skip instead of omit, otherwise fbcode ci can be flaky
        for test_name in skip_list:
            test_failures_gpu_wrapper[f"{test_name}_cuda"] = (
                test_torchinductor.TestFailure(("gpu_wrapper",), is_skip=True)
            )
            test_failures_gpu_wrapper[f"{test_name}_gpu_dynamic_shapes"] = (
                test_torchinductor.TestFailure(("gpu_wrapper",), is_skip=True)
            )

    test_torchinductor.copy_tests(
        GpuWrapperTemplate, TestGpuWrapper, "gpu_wrapper", test_failures_gpu_wrapper
    )

    DynamicShapesGpuWrapperTemplate = (
        test_torchinductor_dynamic_shapes.make_dynamic_cls(GpuWrapperTemplate)
    )

    test_torchinductor.copy_tests(
        DynamicShapesGpuWrapperTemplate,
        DynamicShapesGpuWrapperGpuTests,
        "gpu_wrapper",
        test_failures_gpu_wrapper,
        xfail_prop="_expected_failure_dynamic_wrapper",
    )

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_GPU:
        run_tests(needs="filelock")
