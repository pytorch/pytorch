# Owner(s): ["module: dynamo"]
from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches

try:
    from . import (
        test_aot_autograd,
        test_ctx_manager,
        test_export,
        test_functions,
        test_higher_order_ops,
        test_misc,
        test_modules,
        test_python_autograd,
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
    import test_python_autograd
    import test_repros
    import test_subgraphs

import unittest


test_classes = {}

ALL_DYNAMIC_XFAILS = {}

XFAIL_HITS = 0


def make_test_cls(cls, *, dynamic, static_default=False):
    if dynamic:
        suffix = "_dynamic_shapes"
        cls_prefix = "DynamicShapes"
    else:
        suffix = "_static_shapes"
        cls_prefix = "StaticShapes"

    if static_default:
        suffix += "_static_default"
        cls_prefix = f"StaticDefault{cls_prefix}"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "dynamic_shapes", dynamic),
        (config, "assume_static_by_default", static_default),
        (config, "specialize_int", static_default),
    )

    xfail_tests = ALL_DYNAMIC_XFAILS.get(cls.__name__)
    if xfail_tests is not None:
        global XFAIL_HITS
        XFAIL_HITS += 1
        for t in xfail_tests:
            unittest.expectedFailure(getattr(test_class, f"{t}{suffix}"))

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
    test_aot_autograd.AotAutogradFallbackTests,
    test_python_autograd.TestPythonAutograd,
]
for test in tests:
    make_test_cls(test, dynamic=False)
    make_test_cls(test, dynamic=True, static_default=True)

assert XFAIL_HITS == len(ALL_DYNAMIC_XFAILS) * 2

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
