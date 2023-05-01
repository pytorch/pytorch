# Owner(s): ["module: dynamo"]
from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches

try:
    from . import (
        test_ctx_manager,
        test_export,
        test_functions,
        test_misc,
        test_modules,
        test_repros,
        test_subgraphs,
    )
except ImportError:
    import test_ctx_manager
    import test_export
    import test_functions
    import test_misc
    import test_modules
    import test_repros
    import test_subgraphs

import unittest


test_classes = {}

ALL_DYNAMIC_XFAILS = {
    "MiscTests": [],
}

XFAIL_HITS = 0


def make_dynamic_cls(cls, *, static_default=False):
    suffix = "_dynamic_shapes"
    if static_default:
        suffix += "_static_default"

    cls_prefix = "DynamicShapes"
    if static_default:
        cls_prefix = f"StaticDefault{cls_prefix}"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "dynamic_shapes", True),
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
]
for test in tests:
    make_dynamic_cls(test, static_default=True)

assert XFAIL_HITS == len(ALL_DYNAMIC_XFAILS)

# Single config failures

unittest.expectedFailure(
    StaticDefaultDynamicShapesMiscTests.test_change_backends
    # '__torch__.torch.SymInt (of Python compilation unit at: 0x4c9c0e0)'
    # object has no attribute or method '__ne__'
    # NB: I don't think this ever can actually work, cuz TorchScript
    # can't deal with SymInt inputs
)


unittest.expectedFailure(
    StaticDefaultDynamicShapesMiscTests.test_slice_input
    # NotImplementedError: SymNodeVariable() is not a constant
)

unittest.expectedFailure(
    StaticDefaultDynamicShapesNNModuleTests.test_lazy_module1
    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
)

unittest.expectedFailure(
    StaticDefaultDynamicShapesNNModuleTests.test_lazy_module2
    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
)

unittest.expectedFailure(
    StaticDefaultDynamicShapesNNModuleTests.test_lazy_module3
    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
)

unittest.expectedFailure(
    StaticDefaultDynamicShapesNNModuleTests.test_lazy_module4
    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
)

unittest.expectedFailure(
    StaticDefaultDynamicShapesNNModuleTests.test_lazy_module5
    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
)

unittest.expectedFailure(
    StaticDefaultDynamicShapesNNModuleTests.test_lazy_module6
    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
