# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import make_test_cls_with_patches


try:
    from . import (
        test_activation_checkpointing,
        test_comprehensions,
        test_ctx_manager,
        test_decorators,
        test_dicts,
        test_exceptions,
        test_functions,
        test_global,
        test_hooks,
        test_list,
        test_misc,
        test_modules,
        test_recompiles,
        test_repros,
        test_subgraphs,
        test_unspec,
    )
except ImportError:
    import test_activation_checkpointing
    import test_comprehensions
    import test_ctx_manager
    import test_decorators
    import test_dicts
    import test_exceptions
    import test_functions
    import test_global
    import test_hooks
    import test_list
    import test_misc

    import test_modules
    import test_recompiles
    import test_repros
    import test_subgraphs
    import test_unspec


test_classes = {}


def make_nested_cls(cls):
    config = torch._dynamo.config

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
    test_comprehensions.ComprehensionTests,
    test_ctx_manager.CtxManagerTests,
    test_decorators.DecoratorTests,
    test_dicts.DictTests,
    test_exceptions.ExceptionTests,
    test_functions.FunctionTests,
    test_functions.DefaultsTests,
    test_global.TestGlobals,
    test_hooks.HooksTests,
    test_list.TupleTests,
    test_misc.MiscTests,
    test_modules.NNModuleTests,
    test_recompiles.RecompileTests,
    test_repros.ReproTests,
    test_subgraphs.SubGraphTests,
    test_unspec.UnspecTests,
]

test = None
for test in tests:
    if not test:
        continue
    make_nested_cls(test)

del test

xfails = [
    # variable naming issues (debug_force_nested_calls changes L['x'] to L['args'][0])
    NestedGraphBreaksMiscTests.test_flat_name_to_original_fqn_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_compare_shapes_with_constant_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_guard_failure_fn2_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_guard_failure_fn_shape_control_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_guard_filter_fn_by_name_and_value_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_guard_sym_node_fstring_when_used_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_replay_side_effects_config_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_replay_side_effects_model_attr_nested_graph_breaks,  # noqa: F821
    # doesn't work due to debug_force_nested_calls wrapping the top frame
    NestedGraphBreaksMiscTests.test_dynamo_cache_move_to_front_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_dynamo_reset_clears_cache_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_fail_on_recompile_error_message_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_get_cache_entry_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_precompile_entries_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_precompile_entry_hit_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_precompile_fail_on_recompile_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_torch_guards_stack_frame_register_inlining_deep_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksMiscTests.test_torch_guards_stack_frame_register_inlining_nested_graph_breaks,  # noqa: F821
    # differing op_count
    NestedGraphBreaksMiscTests.test_nested_closure_nested_graph_breaks,  # noqa: F821
    # debug_force_nested_calls tries to inline skipped functions
    NestedGraphBreaksTupleTests.test_binop_add_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksTupleTests.test_binop_imul_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksTupleTests.test_cmp_eq_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksTupleTests.test_cmp_greater_than_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksTupleTests.test_cmp_greater_than_or_equal_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksTupleTests.test_cmp_less_than_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksTupleTests.test_cmp_less_than_or_equal_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksTupleTests.test_cmp_ne_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksTupleTests.test___contains___nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksTupleTests.test_count_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksTupleTests.test___getitem___nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksTupleTests.test_index_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksTupleTests.test___iter___nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksTupleTests.test_list_mul_constant_tuple_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksFunctionTests.test_itertools_islice_basic_ops_nested_graph_breaks,  # noqa: F821
    # bytecode codegen issues with nested graph breaks
    NestedGraphBreaksDefaultsTests.test_frozenset_reconstruction2_nested_graph_breaks,  # noqa: F821
    # correctness issues due to debug_force_nested_calls wrapping
    NestedGraphBreaksDecoratorTests.test_fullgraph_eval_frame_override_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksDecoratorTests.test_torch_guards_stack_frame_register_inlining_disable_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksSubGraphTests.test_resume_paths_join_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksReproTests.test_udf_classes_reconstruction_nested_graph_breaks,  # noqa: F821
    NestedGraphBreaksUnspecTests.test_unspecialized_float_multiply_precision,  # noqa: F821
]

case = None

for case in xfails:
    unittest.expectedFailure(case)

del case, xfails

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
