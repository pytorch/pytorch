# Owner(s): ["module: export"]

from torch._dynamo.testing import make_test_cls_with_patches
from torch.export import config

try:
    from . import test_export
except ImportError:
    import test_export

test_classes = {}


def make_dynamic_cls(cls):
    suffix = "_non_strict"

    cls_prefix = "NonStrictExport"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "strict_mode_default", False),
        xfail_prop="_expected_failure_non_strict",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_export.TestDynamismExpression,
    test_export.TestExport,
]
for test in tests:
    make_dynamic_cls(test)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
