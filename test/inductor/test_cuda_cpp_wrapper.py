# Owner(s): ["module: inductor"]
import sys
import unittest
from typing import NamedTuple

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.common_device_type import (
    get_desired_device_type_test_bases,
)
from torch.testing._internal.common_utils import slowTest, TEST_WITH_ASAN
from torch.testing._internal.inductor_utils import HAS_CUDA


try:
    try:
        from . import (
            test_foreach,
            test_pattern_matcher,
            test_select_algorithm,
            test_torchinductor,
            test_torchinductor_dynamic_shapes,
        )
    except ImportError:
        import test_foreach
        import test_pattern_matcher
        import test_select_algorithm
        import test_torchinductor
        import test_torchinductor_dynamic_shapes
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


_desired_test_bases = get_desired_device_type_test_bases()
RUN_CUDA = (
    HAS_CUDA
    and any(getattr(x, "device_type", "") == "cuda" for x in _desired_test_bases)
    and not TEST_WITH_ASAN
)


class CudaWrapperTemplate:
    pass


class TestCudaWrapper(InductorTestCase):
    device = "cuda"


class DynamicShapesCudaWrapperCudaTests(InductorTestCase):
    device = "cuda"


test_failures_cuda_wrapper = {
    "test_mm_plus_mm2_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",), is_skip=True
    ),
}


if config.abi_compatible:
    xfail_list = [
        "test_profiler_mark_wrapper_call_cuda",
    ]
    for test_name in xfail_list:
        test_failures_cuda_wrapper[test_name] = test_torchinductor.TestFailure(
            ("cuda_wrapper",), is_skip=False
        )
        test_failures_cuda_wrapper[
            f"{test_name}_dynamic_shapes"
        ] = test_torchinductor.TestFailure(("cuda_wrapper",), is_skip=False)
    skip_list = []
    for test_name in skip_list:
        test_failures_cuda_wrapper[test_name] = test_torchinductor.TestFailure(
            ("cuda_wrapper",), is_skip=True
        )
        test_failures_cuda_wrapper[
            f"{test_name}_dynamic_shapes"
        ] = test_torchinductor.TestFailure(("cuda_wrapper",), is_skip=True)


def make_test_case(
    name,
    device,
    tests,
    condition=True,
    slow=False,
    func_inputs=None,
    code_string_count=None,
):
    test_name = f"{name}_{device}" if device else name
    if code_string_count is None:
        code_string_count = {}

    func = getattr(tests, test_name)
    assert callable(func), "not a callable"
    func = slowTest(func) if slow else func

    @config.patch(cpp_wrapper=True, search_autotune_cache=False)
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
            CudaWrapperTemplate,
            test_name,
            fn,
        )


if RUN_CUDA:

    class BaseTest(NamedTuple):
        name: str
        device: str = "cuda"
        tests: InductorTestCase = test_torchinductor.GPUTests()

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
        BaseTest(
            "test_foreach_cpp_wrapper",
            tests=test_foreach.ForeachTests(),
        ),  # test foreach
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
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
        BaseTest(
            "test_mm_plus_mm3",
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
        BaseTest("test_fft_real_input"),
        BaseTest("test_fft_real_input_real_output"),
        BaseTest("test_dtypeview"),
        BaseTest("test_dtypeview_fusion"),
    ]:
        make_test_case(item.name, item.device, item.tests)

    from torch._inductor.utils import is_big_gpu

    if is_big_gpu(0):
        for item in [
            BaseTest(
                "test_addmm",
                tests=test_select_algorithm.TestSelectAlgorithm(),
            ),
            BaseTest(
                "test_linear_relu",
                tests=test_select_algorithm.TestSelectAlgorithm(),
            ),
        ]:
            make_test_case(item.name, item.device, item.tests)

    test_torchinductor.copy_tests(
        CudaWrapperTemplate, TestCudaWrapper, "cuda_wrapper", test_failures_cuda_wrapper
    )

    DynamicShapesCudaWrapperTemplate = (
        test_torchinductor_dynamic_shapes.make_dynamic_cls(CudaWrapperTemplate)
    )

    test_torchinductor.copy_tests(
        DynamicShapesCudaWrapperTemplate,
        DynamicShapesCudaWrapperCudaTests,
        "cuda_wrapper",
        test_failures_cuda_wrapper,
        xfail_prop="_expected_failure_dynamic_wrapper",
    )

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_CUDA:
        run_tests(needs="filelock")
