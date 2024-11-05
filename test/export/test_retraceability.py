# Owner(s): ["oncall: export"]

try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch.export import export


test_classes = {}


def mocked_retraceability_export_strict(*args, **kwargs):
    ep = export(*args, **kwargs)
    if "dynamic_shapes" in kwargs:
        if isinstance(kwargs["dynamic_shapes"], dict):
            kwargs["dynamic_shapes"] = tuple(kwargs["dynamic_shapes"].values())

    ep = export(ep.module(), *(args[1:]), **kwargs)
    return ep


def mocked_retraceability_export_non_strict(*args, **kwargs):
    if "strict" in kwargs:
        ep = export(*args, **kwargs)
    else:
        ep = export(*args, **kwargs, strict=False)
    if "dynamic_shapes" in kwargs:
        if isinstance(kwargs["dynamic_shapes"], dict):
            kwargs["dynamic_shapes"] = tuple(kwargs["dynamic_shapes"].values())

    if "strict" in kwargs:
        ep = export(ep.module(), *(args[1:]), **kwargs)
    else:
        ep = export(ep.module(), *(args[1:]), **kwargs, strict=False)
    return ep


def make_dynamic_cls(cls, strict):
    if strict:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "RetraceExport",
            test_export.RETRACEABILITY_STRICT_SUFFIX,
            mocked_retraceability_export_strict,
            xfail_prop="_expected_failure_retrace",
        )
    else:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "RetraceExportNonStrict",
            test_export.RETRACEABILITY_NON_STRICT_SUFFIX,
            mocked_retraceability_export_non_strict,
            xfail_prop="_expected_failure_retrace_non_strict",
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
    make_dynamic_cls(test, True)
    make_dynamic_cls(test, False)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
