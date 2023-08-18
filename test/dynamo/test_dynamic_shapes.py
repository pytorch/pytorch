# Owner(s): ["module: dynamo"]
import unittest
import warnings

from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches
from torch.testing._internal.common_utils import TEST_Z3

try:
    from . import (
        test_aot_autograd,
        test_ctx_manager,
        test_export,
        test_functions,
        test_higher_order_ops,
        test_misc,
        test_modules,
        test_repros,
        test_subgraphs,
    )
except ImportError:
    import test_aot_autograd
    import test_ctx_manager
    import test_export
    import test_functions
    import test_higher_order_ops
    import test_misc
    import test_modules
    import test_repros
    import test_subgraphs


test_classes = {}


def make_dynamic_cls(cls):
    suffix = "_dynamic_shapes"

    cls_prefix = "DynamicShapes"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "assume_static_by_default", False),
        (config, "specialize_int", False),
        (config, "translation_validation", TEST_Z3),
        xfail_prop="_expected_failure_dynamic",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    return test_class


tests = [
    test_ctx_manager.CtxManagerTests,
    test_functions.FunctionTests,
    test_misc.MiscTests,
    test_repros.ReproTests,
    test_modules.NNModuleTests,
    test_export.ExportTests,
    test_subgraphs.SubGraphTests,
    test_higher_order_ops.HigherOrderOpTests,
    test_higher_order_ops.FuncTorchHigherOrderOpTests,
    test_aot_autograd.AotAutogradFallbackTests,
]
for test in tests:
    make_dynamic_cls(test)

if TEST_Z3:
    # this only fails when z3 is available
    unittest.expectedFailure(
        # SymPy is incorrectly transforming 's0 / 6 == 0.5' into 'False'.
        # Ref: https://github.com/sympy/sympy/issues/25146
        DynamicShapesReproTests.test_dynamic_shapes_float_guard_dynamic_shapes
    )

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not TEST_Z3:
        warnings.warn(
            "translation validation is off. "
            "Testing with translation validation requires Z3."
        )

    run_tests()
