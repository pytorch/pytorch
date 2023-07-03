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


def make_dynamic_cls(cls, automatic_dynamic_shapes=False):
    suffix = "_dynamic_shapes"
    if automatic_dynamic_shapes:
        suffix = "_automatic_dynamic_shapes"

    cls_prefix = "DynamicShapes"
    if automatic_dynamic_shapes:
        cls_prefix = "AutomaticDynamicShapes"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "assume_static_by_default", automatic_dynamic_shapes),
        (config, "automatic_dynamic_shapes", automatic_dynamic_shapes),
        (config, "specialize_int", False),
        xfail_prop="_expected_failure_automatic_dynamic"
        if automatic_dynamic_shapes
        else "_expected_failure_dynamic",
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
    test_aot_autograd.AotAutogradFallbackTests,
]
for test in tests:
    make_dynamic_cls(test)
    make_dynamic_cls(test, automatic_dynamic_shapes=True)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
