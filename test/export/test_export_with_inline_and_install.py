# Owner(s): ["oncall: export"]


from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches


try:
    from . import test_export
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library


test_classes = {}


def make_dynamic_cls(cls):
    suffix = "_inline_and_install"

    cls_prefix = "InlineAndInstall"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "install_params_as_graph_attr", True),
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

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
