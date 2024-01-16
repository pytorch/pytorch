# Owner(s): ["module: inductor"]
import itertools
import sys
import unittest
from typing import NamedTuple

from torch._inductor import config
from torch.testing._internal.common_device_type import (
    get_desired_device_type_test_bases,
)
from torch.testing._internal.common_utils import (
    slowTest,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase as TorchTestCase,
)
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


class TestCudaWrapper(TorchTestCase):
    device = "cuda"


class DynamicShapesCudaWrapperCudaTests(TorchTestCase):
    device = "cuda"


test_failures_cuda_wrapper = {
    "test_add_complex_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_add_inplace_permuted_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_buffer_use_after_remove_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_complex_fallback_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_data_type_propogation_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_dropout2_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_dropout3_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_float_index_expression_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_forced_buffer_realize_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_functionalize_rng_wrappers_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_gather2_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_getitem_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_inductor_layout_optimization_input_mutations_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_inplace_add_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_inplace_resize_as_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_input_mutation1_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_input_mutation3_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_input_mutation5_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_linear_mixed_dtype_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_list_clearing_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_mm_mixed_dtype_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_philox_rand_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_pow3_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_rand_like_deterministic_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_randint_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_randint_kernel_count_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_scheduler_vertical_fusion1_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_slice_mutation2_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_slice_view_with_graph_break_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_split_with_sizes_failed_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_uint4x2_mixed_mm_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_unspec_inputs_cuda": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_AllenaiLongformerBase_repro_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_add_complex_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_add_inplace_permuted_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_addmm_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_buffer_use_after_remove_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_complex_fallback_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_conv_inference_heuristics_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_conv2d_backward_channels_last_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",), is_skip=True  # Segfault
    ),
    "test_data_type_propogation_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_dropout2_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_dropout3_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_float_index_expression_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_forced_buffer_realize_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_functionalize_rng_wrappers_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_gather2_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_getitem_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_inductor_layout_optimization_input_mutations_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_inplace_add_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_inplace_resize_as_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_input_mutation1_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_input_mutation3_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_input_mutation5_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_linear_mixed_dtype_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_linear_relu_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_list_clearing_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_mm_plus_mm2_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_mixed_mm_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",), is_skip=True  # Segfault
    ),
    "test_mixed_mm2_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",), is_skip=True  # Segfault
    ),
    "test_mm_mixed_dtype_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_philox_rand_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_pow3_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_rand_like_deterministic_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_randint_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_randint_kernel_count_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_scheduler_vertical_fusion1_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_slice_mutation2_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_slice_view_with_graph_break_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_split_with_sizes_failed_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_uint4x2_mixed_mm_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
    "test_unspec_inputs_cuda_dynamic_shapes": test_torchinductor.TestFailure(
        ("cuda_wrapper",),
    ),
}

if TEST_WITH_ROCM:
    # Current skips for ROCm - mostly all Tensor-likes failures, need to undergo investigation.
    rocm_exclude_list = [
        "test_addmm_cuda",
        "test_batch_norm_2d_2_cuda",
        "test_bmm1_cuda",
        "test_cat_cuda",
        "test_cat_slice_cat_cuda",
        "test_custom_op_cuda",
        "test_convolution1_cuda",
        "test_foreach_cpp_wrapper_cuda",
        "test_index_put_deterministic_fallback_cuda",
        "test_index_tensor_cuda",
        "test_linear_relu_cuda",
        "test_multi_device_cuda",
        "test_mm_plus_mm2_cuda",
        "test_sum_dtype_cuda",
        "test_transpose_cuda",
    ]

    # Create skip entries for both the cuda and cuda_dynamic_shapes variants
    for test_name in rocm_exclude_list:
        dynamic_shapes_test_name = f"{test_name}_dynamic_shapes"
        test_failures_cuda_wrapper[test_name] = test_torchinductor.TestFailure(
            ("cuda_wrapper",), is_skip=True
        )
        test_failures_cuda_wrapper[
            dynamic_shapes_test_name
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
        tests: TorchTestCase = test_torchinductor.CudaTests()

    # Maintain two separate test lists for cuda and cpp for now
    for item in itertools.chain(
        [
            BaseTest(name)
            for name in dir(test_torchinductor.CommonTemplate)
            if name.startswith("test_")
        ],
        [
            BaseTest(
                "test_foreach_cpp_wrapper",
                tests=test_foreach.ForeachTests(),
            ),  # test foreach
            BaseTest(
                "test_cat_slice_cat",
                tests=test_pattern_matcher.TestPatternMatcher(),
            ),
            BaseTest(
                "test_addmm",
                tests=test_select_algorithm.TestSelectAlgorithm(),
            ),
            BaseTest(
                "test_linear_relu",
                tests=test_select_algorithm.TestSelectAlgorithm(),
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
        ],
    ):
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
    )

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if RUN_CUDA:
        run_tests(needs="filelock")
