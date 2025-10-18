# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import make_test_cls_with_patches


try:
    from . import test_activation_checkpointing, test_ctx_manager, test_misc
except ImportError:
    import test_activation_checkpointing
    import test_ctx_manager
    import test_misc


test_classes = {}


def make_nested_cls(cls, strong):
    config = torch._dynamo.config

    if strong:
        # A strong nested graph break test - will graph break at every leaf function's return
        test_class = make_test_cls_with_patches(
            cls,
            "NestedGraphBreaksStrong",
            "_nested_graph_breaks_strong",
            (config, "nested_graph_breaks", True),
            (config, "debug_force_nested_calls", True),
            (config, "debug_force_graph_break_on_leaf_return", True),
            (config, "debug_disable_compile_counter", True),
            xfail_prop="_expected_failure_nested_graph_breaks_strong",
        )
    else:
        test_class = make_test_cls_with_patches(
            cls,
            "NestedGraphBreaks",
            "_nested_graph_breaks",
            (config, "nested_graph_breaks", True),
            (config, "debug_force_nested_calls", True),
            (config, "debug_disable_compile_counter", True),
            xfail_prop="_expected_failure_nested_graph_breaks",
        )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__


tests = [
    getattr(
        test_activation_checkpointing, "ActivationCheckpointingViaTagsTestsCUDA", None
    ),
    test_ctx_manager.CtxManagerTests,
    test_misc.MiscTests,
]

strong_tests = []
test = None
for test in tests:
    if not test:
        continue
    make_nested_cls(test, False)

for test in strong_tests:
    make_nested_cls(test, True)

del test

xfails = [
    # multiple exit due to nested graph break in decorator
    # NestedGraphBreaksStrongCtxManagerTests.test_disable_saved_tensors_hooks_nested_graph_breaks_strong,  # noqa: F821
    # NestedGraphBreaksStrongCtxManagerTests.test_disable_saved_tensors_hooks_prev_disabled_nested_graph_breaks_strong,  # noqa: F821
    # NestedGraphBreaksStrongCtxManagerTests.test_disable_saved_tensors_hooks_prev_disabled_nested_nested_graph_breaks_strong,  # noqa: F821
    # graph break in context manager __init__
    # NestedGraphBreaksStrongCtxManagerTests.test_generic_context_manager_CustomizedCtxManager_nested_graph_breaks_strong,  # noqa: F821
    # NestedGraphBreaksStrongCtxManagerTests.test_generic_context_manager_customized_ctx_manager_nested_graph_breaks_strong,  # noqa: F821
    # NestedGraphBreaksStrongCtxManagerTests.test_generic_context_manager_with_graph_break_CustomizedCtxManager_nested_graph_breaks_strong,  # noqa: F821
    # NestedGraphBreaksStrongCtxManagerTests.test_generic_context_manager_with_graph_break_customized_ctx_manager_nested_graph_breaks_strong,  # noqa: F821
    # NestedGraphBreaksStrongCtxManagerTests.test_generic_ctx_manager_with_graph_break_CustomizedCtxManagerWithGraphBreak_nested_graph_breaks_strong,  # noqa: F821
    # NestedGraphBreaksStrongCtxManagerTests.test_generic_ctx_manager_with_graph_break_customized_ctx_manager_with_graph_break_nested_graph_breaks_strong,  # noqa: F821
    # NestedGraphBreaksStrongCtxManagerTests.test_nested_generic_context_manager_CustomizedCtxManager_nested_graph_breaks_strong,  # noqa: F821
    # NestedGraphBreaksStrongCtxManagerTests.test_nested_generic_context_manager_customized_ctx_manager_nested_graph_breaks_strong,  # noqa: F821
    # NestedGraphBreaksStrongCtxManagerTests.test_nested_generic_context_manager_with_graph_break_CustomizedCtxManager_nested_graph_breaks_strong,  # noqa: F821
    # NestedGraphBreaksStrongCtxManagerTests.test_nested_generic_context_manager_with_graph_break_customized_ctx_manager_nested_graph_breaks_strong,  # noqa: F821
    # NestedGraphBreaksStrongCtxManagerTests.test_return_context_manager_nested_graph_breaks_strong,  # noqa: F821
    # NestedGraphBreaksStrongCtxManagerTests.test_return_context_manager_with_graph_break_nested_graph_breaks_strong,  # noqa: F821
    # recursion limit exceeded
    # NestedGraphBreaksStrongCtxManagerTests.test_cuda_stream_compared_with_constant_nested_graph_breaks_strong,  # noqa: F821
    # variable naming issues
    NestedGraphBreaksMiscTests.test_flat_name_to_original_fqn_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_compare_shapes_with_constant_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_guard_failure_fn2_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_guard_failure_fn_shape_control_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_guard_failure_fn_tensor_iter_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_guard_filter_fn_by_name_and_value_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_guard_sym_node_fstring_when_used_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_symint_as_device_kwarg_multi_gpu_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_sys_modules_nested_graph_breaks,  # noqa: F821
    # counters["graph_breaks"] issues
    NestedGraphBreaksMiscTests.test_data_ptr_graph_break_aten_nested_graph_breaks,  # noqa: F821
    # nested graph break removes duplicate graph break
    NestedGraphBreaksMiscTests.test_duplicate_graph_break_log_nested_graph_breaks,  # noqa: F821
    # doesn't work due to debug_force_nested_calls wrapping the top frame
    NestedGraphBreaksMiscTests.test_dynamo_cache_invalidate_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_dynamo_cache_move_to_front_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_dynamo_reset_clears_cache_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_fail_on_recompile_error_message_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_get_cache_entry_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_getattrvariable_as_python_constant_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_precompile_entries_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_precompile_entry_hit_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_precompile_fail_on_recompile_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_torch_guards_stack_frame_register_inlining_deep_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_torch_guards_stack_frame_register_inlining_nested_graph_breaks,  # noqa: F821
    # differing op_count
    NestedGraphBreaksMiscTests.test_nested_closure_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_return_nested_function_nested_graph_breaks,  # noqa: F821
    # unknown
    NestedGraphBreaksMiscTests.test_inspect_signature_bind_non_user_function_nested_graph_breaks,  # noqa: F821
]

case = None

for case in xfails:
    unittest.expectedFailure(case)

del case, xfails

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
