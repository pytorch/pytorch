# Owner(s): ["module: inductor"]
import importlib
import os
import re
import sys
import unittest

import torch
from torch._inductor.compile_fx import compile_fx
from torch.testing._internal.common_utils import (
    IS_CI,
    IS_WINDOWS,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor_codegen_dynamic_shapes yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

importlib.import_module("filelock")

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from inductor.test_torchinductor import (
    CommonTemplate,
    copy_tests,
    run_and_get_cpp_code,
    run_and_get_triton_code,
    TestFailure,
)
from inductor.test_torchinductor_dynamic_shapes import make_dynamic_cls


# Checks for patterns in generated C++/Triton code to see if it's dynamic
def check_codegen(
    self: TestCase,
    model,
    example_inputs,
    kwargs=None,
    *,
    is_cpp_code: bool,
):
    kwargs = kwargs or {}

    if is_cpp_code is False:
        if hasattr(model, "to"):
            model = model.to("cuda")

        def copy_fn(x):
            # preserve strides of the input on the device
            if not isinstance(x, torch.Tensor):
                return x
            return torch.empty_strided(
                x.size(), x.stride(), device="cuda", dtype=x.dtype
            ).copy_(x)

        example_inputs = tuple(copy_fn(x) for x in example_inputs)

    torch._dynamo.reset()

    called = False

    def compile_fx_wrapper(model_, example_inputs_):
        nonlocal called
        called = True
        return compile_fx(model_, example_inputs_)

    def run(*ex, **kwargs):
        return model(*ex, **kwargs)

    run = torch._dynamo.optimize(compile_fx_wrapper, nopython=True)(run)

    if is_cpp_code:
        code = run_and_get_cpp_code(run, *example_inputs, **kwargs)
        for_loop_found = False
        has_dynamic = False
        lines = code.split("\n")
        for line in lines:
            if "for(" in line:
                for_loop_found = True
                if re.search(r";.*ks.*;", line) is not None:
                    has_dynamic = True
                    break
        self.assertTrue(
            has_dynamic, msg=f"Failed to find dynamic for loop variable\n{code}"
        )
        self.assertTrue(for_loop_found, f"Failed to find for loop\n{code}")
    else:
        code = run_and_get_triton_code(run, *example_inputs, **kwargs)
        triton_kernel_found = False
        lines = code.split("\n")
        for line in lines:
            if "def triton" in line:
                triton_kernel_found = True
                continue
        self.assertTrue(triton_kernel_found, f"Failed to find triton kernel\n{code}")

    assert called, "Ran graph without calling compile_fx"

    torch._dynamo.reset()


# xfail by default, set is_skip=True to skip
test_failures = {
    #
    # Failed to find dynamic for loop variable:
    #
    "test_arange1_dynamic_shapes": TestFailure(("cpu",)),
    "test_arange2_dynamic_shapes": TestFailure(("cpu",)),
    "test_arange3_dynamic_shapes": TestFailure(("cpu",)),
    "test_arange4_dynamic_shapes": TestFailure(("cpu",)),
    "test_arange6_dynamic_shapes": TestFailure(("cpu",)),
    "test_clamp_type_promotion_dynamic_shapes": TestFailure(("cpu",)),
    "test_conv2d_channels_last_dynamic_shapes": TestFailure(("cpu",)),
    "test_conv3d_channels_last_dynamic_shapes": TestFailure(("cpu",)),
    "test_expand_dynamic_shapes": TestFailure(("cpu",)),
    "test_glu_dynamic_shapes": TestFailure(("cpu",)),
    "test_isinf2_dynamic_shapes": TestFailure(("cpu",)),
    "test_layer_norm_dynamic_shapes": TestFailure(("cuda")),
    "test_linspace1_dynamic_shapes": TestFailure(("cpu",)),
    "test_reflection_pad2d_backward_dynamic_shapes": TestFailure(("cpu",)),
    "test_reflection_pad2d_dynamic_shapes": TestFailure(("cpu",)),
    "test_stack_dynamic_shapes": TestFailure(("cpu",)),
    "test_tensor2_dynamic_shapes": TestFailure(("cpu",)),
    "test_tensor3_dynamic_shapes": TestFailure(("cpu",)),
    "test_to_device_constant_dynamic_shapes": TestFailure(("cpu")),
    "test_upsample_nearest2d_backward_dynamic_shapes": TestFailure(("cpu",)),
    "test_views3_dynamic_shapes": TestFailure(("cpu",)),
    "test_views4_dynamic_shapes": TestFailure(("cpu",)),
    "test_zeros_dynamic_shapes": TestFailure(("cpu",)),
    "test_uint_dynamic_shapes": TestFailure(("cpu",)),
    "test_issue102546_dynamic_shapes": TestFailure(("cpu",)),
    #
    # Failed to find for loop/triton kernel:
    #
    "test_complex_fallback_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_adaptive_avg_pool2d2_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_argmax_to_float_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_avg_pool2d7_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_avg_pool2d_backward4_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_baddbmm_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_bmm2_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_both_scalars_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_compar_dynamic_shapes": TestFailure(("cpu",)),
    "test_const_int32_to_float_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_conv2d_backward_channels_last_dynamic_shapes": TestFailure(("cpu",)),
    "test_conv_backward_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_conv_functional_bn_fuse_dynamic_shapes": TestFailure(("cpu",), is_skip=True),
    "test_convolution2_dynamic_shapes": TestFailure(("cpu",)),
    "test_div8_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_embedding_bag_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_empty1_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_empty2_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_empty_strided_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_index3_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_inductor_bucketize_dynamic_shapes": TestFailure(("cpu")),
    "test_inductor_bucketize_default_kwargs_dynamic_shapes": TestFailure(("cpu")),
    "test_inductor_bucketize_int_dynamic_shapes": TestFailure(("cpu")),
    "test_like_rands_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_linspace2_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_linspace3_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_max_pool2d6_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_max_pool2d8_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_max_pool2d_with_indices_backward5_dynamic_shapes": TestFailure(
        ("cpu", "cuda")
    ),
    "test_max_pool2d_with_indices_backward6_dynamic_shapes": TestFailure(
        ("cpu", "cuda")
    ),
    "test_misaligned_address_issue1_dynamic_shapes": TestFailure(("cpu",)),
    "test_mm_views_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_new_empty_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_new_empty_strided_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_new_ones_dynamic_shapes": TestFailure(("cpu",)),
    "test_permute2_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_randn_generator_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_randn_like_empty_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_single_elem_dynamic_shapes": TestFailure(("cpu",)),
    "test_single_elem_indirect_dynamic_shapes": TestFailure(("cpu",)),
    "test_sort_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_split_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_to_device_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_topk_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_unbind_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_views5_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_views6_dynamic_shapes": TestFailure(("cpu",)),
    "test_view_detach_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_view_on_aliased_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_linear_float64_dynamic_shapes": TestFailure(("cpu")),
    #
    # Tests not using 'common' or directly calling 'assertEqual':
    #
    "test_arange5_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_cat_inplace_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_cat_of_loops_and_extern_kernel_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_cauchy_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_scaled_dot_product_efficient_attention_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_dropout_deterministic_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_dropout_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_dtype_mismatch_issue_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_forced_buffer_realize_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_gather2_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_gather_scatter_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_inplace_add_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_inplace_mixed_dtype_ops_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_input_mutation1_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_input_mutation2_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_input_mutation3_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_input_mutation4_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_kernel_names_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_lerp_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_linear_buffer_reuse_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_list_clearing_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_dropout2_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_dropout3_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_masked_fill_promotion_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_min_max_reduction_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_move_arange_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_multi_gpu_recompile_on_index_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_output_strides_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_pow3_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_profiler_mark_wrapper_call_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_rand_like_deterministic_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_scheduler_vertical_fusion1_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_slice_mutation2_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_softmax_one_kernel_loop_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_softmax_one_kernel_persist_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_strided_inputs_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_transpose_add_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_transposed_propagates_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    "test_unspec_inputs_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_vertical_fusion1_dynamic_shapes": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_zero_dim_reductions_dynamic_shapes": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    #
    # The following tests do not support dynamic shapes yet:
    #
    "test_cudnn_rnn_dynamic_shapes": TestFailure(("cuda",)),
    "test_kwargs_dynamic_shapes": TestFailure(("cpu",)),
    "test_fft_real_input_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_fft_real_input_real_output_dynamic_shapes": TestFailure(("cpu", "cuda")),
    # test_roi_align uses torchvision, which doesn't work with dynamic shapes
    "test_roi_align_dynamic_shapes": TestFailure(("cpu", "cuda")),
    "test_aliased_buffer_reuse_dynamic_shapes": TestFailure(("cpu",)),
}

if TEST_WITH_ROCM:
    # aten.miopen_batch_norm is not registered for lowering
    test_failures["test_batch_norm_2d_dynamic_shapes"] = TestFailure(("cuda"))

DynamicShapesCodegenCommonTemplate = make_dynamic_cls(
    CommonTemplate, xfail_prop="_expected_failure_codegen_dynamic"
)


if HAS_CPU:

    class DynamicShapesCodegenCpuTests(TestCase):
        maxDiff = None
        device = "cpu"

        def common(self: TestCase, model, example_inputs, kwargs=None, **_rest):
            return check_codegen(
                self=self,
                model=model,
                example_inputs=example_inputs,
                kwargs=kwargs,
                is_cpp_code=True,
            )

    copy_tests(
        DynamicShapesCodegenCommonTemplate,
        DynamicShapesCodegenCpuTests,
        "cpu",
        test_failures,
    )


if HAS_CUDA and not TEST_WITH_ASAN:

    class DynamicShapesCodegenCudaTests(TestCase):
        maxDiff = None
        device = "cuda"

        def common(self: TestCase, model, example_inputs, kwargs=None, **_rest):
            return check_codegen(
                self=self,
                model=model,
                example_inputs=example_inputs,
                kwargs=kwargs,
                is_cpp_code=False,
            )

    copy_tests(
        DynamicShapesCodegenCommonTemplate,
        DynamicShapesCodegenCudaTests,
        "cuda",
        test_failures,
    )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU or HAS_CUDA:
        run_tests(needs="filelock")
