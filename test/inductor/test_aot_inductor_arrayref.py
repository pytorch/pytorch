# Owner(s): ["module: inductor"]
import sys
import unittest

from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_CI, IS_FBCODE, IS_WINDOWS


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    try:
        from .test_aot_inductor import (
            AOTInductorTestsTemplate,
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from .test_torchinductor import copy_tests, TestFailure
    except ImportError:
        from test_aot_inductor import (  # @manual
            AOTInductorTestsTemplate,
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            copy_tests,
            TestFailure,
        )
except (unittest.SkipTest, ImportError) as e:
    if __name__ == "__main__":
        sys.exit(0)
    raise


def fail_stack_allocation(is_skip=False):
    return TestFailure(
        (
            "cpu_with_stack_allocation",
            "cpu_with_stack_allocation_and_minimal_arrayref_interface",
        ),
        is_skip=is_skip,
    )


def fail_minimal_arrayref_interface(is_skip=False):
    return TestFailure(
        ("cpu_with_stack_allocation_and_minimal_arrayref_interface",),
        is_skip=is_skip,
    )


# test_failures, xfail by default, set is_skip=True to skip
CPU_TEST_FAILURES = {
    # TODO: error: ‘complex64’ was not declared in this scope
    "test_add_complex": fail_minimal_arrayref_interface(is_skip=True),
    "test_conv_freezing": fail_minimal_arrayref_interface(is_skip=True),
    "test_deconv_freezing": fail_minimal_arrayref_interface(is_skip=True),
    "test_cond_nested": fail_minimal_arrayref_interface(),
    "test_cond_simple": fail_minimal_arrayref_interface(),
    "test_cond_symint_input": fail_minimal_arrayref_interface(),
    "test_cond_use_buffers_from_outer_scope": fail_minimal_arrayref_interface(),
    "test_cond_with_multiple_outputs": fail_minimal_arrayref_interface(),
    "test_cond_with_parameters": fail_minimal_arrayref_interface(),
    "test_cond_with_reinterpret_view_inputs_outputs": fail_minimal_arrayref_interface(),
    "test_cond_share_predicte": fail_stack_allocation(is_skip=True),
    "test_while_loop_with_parameters": fail_minimal_arrayref_interface(),
    "test_while_loop_with_pytree_inputs": fail_stack_allocation(),
    # FIXME: failed with Segfault while exiting the Python runtime
    "test_duplicate_constant_folding": fail_stack_allocation(is_skip=True),
    "test_stride_with_unbacked_expr": fail_minimal_arrayref_interface(is_skip=True),
    # TODO: use of deleted function RAIIAtenTensorHandle
    "test_dup_unbacked_sym_decl": fail_minimal_arrayref_interface(is_skip=True),
    # TODO: use of deleted function RAIIAtenTensorHandle
    "test_dup_unbacked_sym_decl_with_refinement": fail_minimal_arrayref_interface(
        is_skip=True
    ),
    # https://github.com/pytorch/pytorch/issues/129550
    # https://github.com/pytorch/pytorch/issues/123691
    "test_dynamic_scalar": fail_minimal_arrayref_interface(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/122980
    "test_fft_c2c": fail_stack_allocation(is_skip=True),
    "test_freezing": fail_minimal_arrayref_interface(is_skip=True),
    "test_linear_freezing": fail_minimal_arrayref_interface(is_skip=True),
    # FIXME: failed with Segfault while exiting the Python runtime
    "test_missing_cubin": fail_stack_allocation(is_skip=True),
    # minimal arrayref interface only works with CPU; test crashes.
    # https://github.com/pytorch/pytorch/issues/122983
    "test_multi_device": fail_minimal_arrayref_interface(is_skip=True),
    # TODO: AssertionError: unsupported Optional type in convert_arg_type: Generator
    "test_normal_functional": fail_stack_allocation(is_skip=True),
    # the test segfaults
    "test_repeat_output": fail_stack_allocation(is_skip=True),
    # segfault
    "test_buffer_mutation_1": fail_stack_allocation(is_skip=True),
    # segfault
    "test_buffer_mutation_2": fail_stack_allocation(is_skip=True),
    # segfault
    "test_bool_input": fail_stack_allocation(is_skip=True),
    # segfault
    "test_int_list_input": fail_stack_allocation(is_skip=True),
    # segfault
    # 'AOTInductorTestABICompatibleCpuWithStackAllocation' object has no attribute 'code_check_count'
    "test_buffer_mutation_3": fail_stack_allocation(is_skip=True),
    "test_zero_size_buffer": fail_stack_allocation(is_skip=True),
    # FIXME: failed with Segfault while exiting the Python runtime
    "test_scatter_fallback": fail_stack_allocation(is_skip=True),
    # Looks like the same issue as https://github.com/pytorch/pytorch/issues/122978
    "test_scatter_reduce_fallback": fail_minimal_arrayref_interface(is_skip=True),
    # Looks like the same issue as https://github.com/pytorch/pytorch/issues/122978
    "test_index_put_fallback": fail_minimal_arrayref_interface(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/122984
    "test_index_put_with_none_index": fail_minimal_arrayref_interface(is_skip=True),
    # FIXME: failed with Segfault while exiting the Python runtime
    "test_constant": fail_stack_allocation(is_skip=True),
    # Looks like the same issue as https://github.com/pytorch/pytorch/issues/122978
    "test_shifted_constraint_ranges": fail_stack_allocation(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/123691
    "test_amp_fallback_random": fail_minimal_arrayref_interface(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/123691
    "test_zero_grid_with_unbacked_symbols": fail_minimal_arrayref_interface(
        is_skip=True
    ),
    # failed on MacOS
    "test_zero_grid_with_backed_symbols": fail_stack_allocation(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/122990
    "test_cond_non_tensor_predicates_dynamic_False": fail_stack_allocation(
        is_skip=True
    ),
    # same issue as https://github.com/pytorch/pytorch/issues/122990
    "test_cond_non_tensor_predicates_dynamic_True": fail_stack_allocation(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/122991
    "test_runtime_checks_complex": fail_stack_allocation(is_skip=True),
    "test_runtime_checks_fp8": fail_stack_allocation(is_skip=True),
    "test_while_loop_simple": fail_stack_allocation(is_skip=True),
    "test_while_loop_nested": fail_stack_allocation(is_skip=True),
    "test_while_loop_with_outer_code": fail_stack_allocation(is_skip=True),
    # TODO: error: cannot convert ArrayRefTensor<float> to AtenTensorHandle
    "test_while_loop_with_outer_buffers": fail_stack_allocation(is_skip=True),
    # TODO: use of undeclared identifier 'float8_e4m3fn' and 'half'
    "test_fp8": fail_minimal_arrayref_interface(is_skip=True),
    "test_size_from_multi_output": fail_stack_allocation(is_skip=True),
    "test_torchvision_transforms_functional_tensor_resize": fail_minimal_arrayref_interface(),
    # TODO: AttributeError: 'ShapeAsConstantBuffer' object has no attribute 'dtype'
    "test_symint_item": fail_minimal_arrayref_interface(is_skip=True),
    # TODO: AttributeError: 'ShapeAsConstantBuffer' object has no attribute 'dtype'
    "test_symbool_item": fail_minimal_arrayref_interface(is_skip=True),
    # TODO: AttributeError: 'ShapeAsConstantBuffer' object has no attribute 'dtype'
    "test_symfloat_item": fail_minimal_arrayref_interface(is_skip=True),
    "test_update_constant_buffer": fail_stack_allocation(is_skip=True),
    "test_so_without_weight": fail_stack_allocation(is_skip=True),
}


class AOTInductorTestABICompatibleCpuWithStackAllocation(TestCase):
    device = "cpu"
    device_type = "cpu"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = True
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCpuWithStackAllocation,
    "cpu_with_stack_allocation",
    CPU_TEST_FAILURES,
)


class AOTInductorTestABICompatibleCpuWithStackAllocationAndMinimalArrayRefInterface(
    TestCase
):
    device = "cpu"
    device_type = "cpu"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = True
    use_minimal_arrayref_interface = True


if IS_FBCODE:
    # The following tests look like they pass in both pytest and unittest (xml
    # and terminal output say pass), but the process will segfault.  This only
    # happens in OSS CI and is fine internally.
    # See https://github.com/pytorch/pytorch/issues/123691
    copy_tests(
        AOTInductorTestsTemplate,
        AOTInductorTestABICompatibleCpuWithStackAllocationAndMinimalArrayRefInterface,
        "cpu_with_stack_allocation_and_minimal_arrayref_interface",
        CPU_TEST_FAILURES,
    )

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests(needs="filelock")
