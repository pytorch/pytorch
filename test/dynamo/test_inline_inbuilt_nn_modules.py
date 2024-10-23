# Owner(s): ["module: dynamo"]
import unittest

from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches


try:
    from . import (
        test_aot_autograd,
        test_functions,
        test_higher_order_ops,
        test_misc,
        test_modules,
    )
except ImportError:
    import test_aot_autograd
    import test_functions
    import test_higher_order_ops
    import test_misc

    import test_modules


test_classes = {}


def make_inline_inbuilt_nn_modules_cls(cls):
    suffix = "_inline_inbuilt_nn_modules"

    cls_prefix = "InlineInbuiltNNModules"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "inline_inbuilt_nn_modules", True),
        xfail_prop="_expected_failure_inline_inbuilt_nn_modules",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_misc.MiscTests,
    test_functions.FunctionTests,
    test_modules.NNModuleTests,
    test_higher_order_ops.HigherOrderOpTests,
    test_higher_order_ops.FuncTorchHigherOrderOpTests,
    test_aot_autograd.AotAutogradFallbackTests,
    # test_repros.ReproTests,
]
for test in tests:
    make_inline_inbuilt_nn_modules_cls(test)
del test

unittest.skip(
    InlineInbuiltNNModulesMiscTests.test_cpp_extension_recommends_custom_ops_inline_inbuilt_nn_modules  # noqa: F821
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
