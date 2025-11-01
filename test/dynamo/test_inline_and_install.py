# Owner(s): ["module: dynamo"]
import unittest

from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches


try:
    from . import test_export
except ImportError:
    import test_export


test_classes = {}


def make_dynamic_cls(cls):
    suffix = "_inline_and_install"

    cls_prefix = "InlineAndInstall"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "install_free_tensors", True),
        (config, "inline_inbuilt_nn_modules", True),
        xfail_prop="_expected_failure_inline_and_install",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_export.ExportTests,
]
for test in tests:
    make_dynamic_cls(test)
del test

# After installing and inlining is turned on, these tests won't throw
# errors in export (which is expected for the test to pass)
# Therefore, these unittest are expected to fail, and we need to update the
# semantics
unittest.expectedFailure(
    InlineAndInstallExportTests.test_invalid_input_global_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallExportTests.test_invalid_input_global_multiple_access_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallExportTests.test_invalid_input_nonlocal_inline_and_install  # noqa: F821
)


# This particular test is marked expecting failure, since dynamo was creating second param for a
# and this was causing a failure in the sum; however with these changes, that test is fixed
# so will now pass, so we need to mark that it is no longer expected to fail
def expectedSuccess(test_item):
    test_item.__unittest_expecting_failure__ = False
    return test_item


expectedSuccess(
    InlineAndInstallExportTests.test_sum_param_inline_and_install  # noqa: F821
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
