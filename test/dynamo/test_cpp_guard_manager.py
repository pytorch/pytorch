# Owner(s): ["module: dynamo"]

from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches

try:
    from . import (
        test_functions,
        test_higher_order_ops,
        test_misc,
        test_optimizers,
        test_repros,
    )
except ImportError:
    import test_functions
    import test_higher_order_ops
    import test_misc
    import test_optimizers
    import test_repros


test_classes = {}


def make_cpp_guard_manager_cls(cls):
    suffix = "_cpp_guard_manager"

    cls_prefix = "CppGuardManager"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "enable_cpp_guard_manager", True),
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_functions.FunctionTests,
    test_misc.MiscTests,
    test_repros.ReproTests,
    test_higher_order_ops.HigherOrderOpTests,
    test_higher_order_ops.FuncTorchHigherOrderOpTests,
    test_optimizers.End2EndTests,
]
for test in tests:
    make_cpp_guard_manager_cls(test)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
