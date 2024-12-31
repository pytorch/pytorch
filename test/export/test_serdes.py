# Owner(s): ["oncall: export"]

import io


try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch.export import export, load, save


test_classes = {}


def mocked_serder_export_strict(*args, **kwargs):
    ep = export(*args, **kwargs)
    buffer = io.BytesIO()
    save(ep, buffer)
    buffer.seek(0)
    loaded_ep = load(buffer)
    return loaded_ep


def mocked_serder_export_non_strict(*args, **kwargs):
    if "strict" in kwargs:
        ep = export(*args, **kwargs)
    else:
        ep = export(*args, **kwargs, strict=False)
    buffer = io.BytesIO()
    save(ep, buffer)
    buffer.seek(0)
    loaded_ep = load(buffer)
    return loaded_ep


def make_dynamic_cls(cls, strict):
    if strict:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "SerDesExport",
            test_export.SERDES_SUFFIX,
            mocked_serder_export_strict,
            xfail_prop="_expected_failure_serdes",
        )
    else:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "SerDesExportNonStrict",
            test_export.SERDES_NON_STRICT_SUFFIX,
            mocked_serder_export_non_strict,
            xfail_prop="_expected_failure_serdes_non_strict",
        )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__


tests = [
    test_export.TestDynamismExpression,
    test_export.TestExport,
]
for test in tests:
    make_dynamic_cls(test, True)
    make_dynamic_cls(test, False)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
