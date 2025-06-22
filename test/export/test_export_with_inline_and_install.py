# Owner(s): ["oncall: export"]


import unittest

from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches


try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch.export import export


test_classes = {}


def mocked_strict_export(*args, **kwargs):
    # If user already specified strict, don't make it strict
    if "strict" in kwargs:
        return export(*args, **kwargs)
    return export(*args, **kwargs, strict=True)


def make_dynamic_cls(cls):
    # Some test check for ending in suffix; need to make
    # the `_strict` for end of string as a result
    suffix = test_export.INLINE_AND_INSTALL_STRICT_SUFFIX

    cls_prefix = "InlineAndInstall"

    cls_a = testing.make_test_cls_with_mocked_export(
        cls,
        "StrictExport",
        suffix,
        mocked_strict_export,
        xfail_prop="_expected_failure_strict",
    )
    test_class = make_test_cls_with_patches(
        cls_a,
        cls_prefix,
        "",
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
    test_export.TestDynamismExpression,
    test_export.TestExport,
]
for test in tests:
    make_dynamic_cls(test)
del test


# NOTE: For this test, we have a failure that occurs because the buffers (for BatchNorm2D) are installed, and not
# graph input.  Therefore, they are not in the `program.graph_signature.inputs_to_buffers`
# and so not found by the unit test when counting the buffers
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_buffer_util_inline_and_install_strict  # noqa: F821
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
