# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import make_test_cls_with_patches


try:
    from . import test_ctx_manager
except ImportError:
    # import test_aot_autograd
    import test_ctx_manager

    # import test_export
    # import test_functions
    # import test_higher_order_ops
    # import test_misc
    # import test_modules
    # import test_repros
    # import test_sdpa
    # import test_subgraphs


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
    # A strong nested graph break test - will graph break at every leaf function's return
    test_class_strong = make_test_cls_with_patches(
        cls,
        "NestedGraphBreaksStrong",
        "_nested_graph_breaks_strong",
        (config, "nested_graph_breaks", True),
        (config, "debug_force_nested_calls", True),
        (config, "debug_force_graph_break_on_leaf_return", True),
        (config, "debug_disable_compile_counter", True),
        xfail_prop="_expected_failure_nested_graph_breaks_strong",
    )

    test_classes[test_class.__name__] = test_class
    test_classes[test_class_strong.__name__] = test_class_strong
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    globals()[test_class_strong.__name__] = test_class_strong
    test_class.__module__ = __name__
    test_class_strong.__module__ = __name__


tests = [
    test_ctx_manager.CtxManagerTests,
    # test_functions.FunctionTests,
    # test_misc.MiscTests,
    # test_repros.ReproTests,
    # test_modules.NNModuleTests,
    # test_subgraphs.SubGraphTests,
    # test_higher_order_ops.HigherOrderOpTests,
    # test_higher_order_ops.FuncTorchHigherOrderOpTests,
    # test_aot_autograd.AotAutogradFallbackTests,
    # test_sdpa.TestSDPA,
]
test = None
for test in tests:
    make_nested_cls(test)
del test

xfails = [
    # multiple exit due to nested graph break in decorator
    NestedGraphBreaksStrongCtxManagerTests.test_disable_saved_tensors_hooks_nested_graph_breaks_strong,  # noqa: F821
    NestedGraphBreaksStrongCtxManagerTests.test_disable_saved_tensors_hooks_prev_disabled_nested_graph_breaks_strong,  # noqa: F821
    NestedGraphBreaksStrongCtxManagerTests.test_disable_saved_tensors_hooks_prev_disabled_nested_nested_graph_breaks_strong,  # noqa: F821
    # graph break in context manager __init__
    NestedGraphBreaksStrongCtxManagerTests.test_generic_context_manager_CustomizedCtxManager_nested_graph_breaks_strong,  # noqa: F821
    NestedGraphBreaksStrongCtxManagerTests.test_generic_context_manager_customized_ctx_manager_nested_graph_breaks_strong,  # noqa: F821
    NestedGraphBreaksStrongCtxManagerTests.test_generic_context_manager_with_graph_break_CustomizedCtxManager_nested_graph_breaks_strong,  # noqa: F821
    NestedGraphBreaksStrongCtxManagerTests.test_generic_context_manager_with_graph_break_customized_ctx_manager_nested_graph_breaks_strong,  # noqa: F821
    NestedGraphBreaksStrongCtxManagerTests.test_generic_ctx_manager_with_graph_break_CustomizedCtxManagerWithGraphBreak_nested_graph_breaks_strong,  # noqa: F821
    NestedGraphBreaksStrongCtxManagerTests.test_generic_ctx_manager_with_graph_break_customized_ctx_manager_with_graph_break_nested_graph_breaks_strong,  # noqa: F821
    NestedGraphBreaksStrongCtxManagerTests.test_nested_generic_context_manager_CustomizedCtxManager_nested_graph_breaks_strong,  # noqa: F821
    NestedGraphBreaksStrongCtxManagerTests.test_nested_generic_context_manager_customized_ctx_manager_nested_graph_breaks_strong,  # noqa: F821
    NestedGraphBreaksStrongCtxManagerTests.test_nested_generic_context_manager_with_graph_break_CustomizedCtxManager_nested_graph_breaks_strong,  # noqa: F821
    NestedGraphBreaksStrongCtxManagerTests.test_nested_generic_context_manager_with_graph_break_customized_ctx_manager_nested_graph_breaks_strong,  # noqa: F821
    NestedGraphBreaksStrongCtxManagerTests.test_return_context_manager_nested_graph_breaks_strong,  # noqa: F821
    NestedGraphBreaksStrongCtxManagerTests.test_return_context_manager_with_graph_break_nested_graph_breaks_strong,  # noqa: F821
    # recursion limit exceeded
    NestedGraphBreaksStrongCtxManagerTests.test_cuda_stream_compared_with_constant_nested_graph_breaks_strong,  # noqa: F821
]

for case in xfails:
    unittest.expectedFailure(case)

del case, xfails

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
