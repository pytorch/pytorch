# Owner(s): ["oncall: export"]

try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch._export import config
from torch.export import export
from torch.testing._internal.common_device_type import instantiate_device_type_tests


test_classes = {}


def mocked_strict_export_v2(*args, **kwargs):
    # If user already specified strict, don't make it strict
    with config.patch(use_legacy_dynamo_graph_capture=False):
        if "strict" in kwargs:
            return export(*args, **kwargs)
        return export(*args, **kwargs, strict=True)


def make_dynamic_cls(cls):
    cls_prefix = "StrictExportV2"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        test_export.STRICT_EXPORT_V2_SUFFIX,
        mocked_strict_export_v2,
        xfail_prop="_expected_failure_strict_v2",
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

for cls, instantiate_kwargs in test_export.DEVICE_EXPORT_TEST_CLASSES:
    instantiate_device_type_tests(
        make_dynamic_cls(cls), globals(), **instantiate_kwargs
    )
del cls, instantiate_kwargs

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
