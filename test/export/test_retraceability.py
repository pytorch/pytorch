# Owner(s): ["oncall: export"]

try:
    from . import test_export, testing
except ImportError:
    import test_export
    import testing
from torch.export import export

test_classes = {}


def mocked_retraceability_export(*args, **kwargs):
    ep = export(*args, **kwargs)
    ep = export(ep.module(), *(args[1:]), **kwargs)
    return ep


def make_dynamic_cls(cls):
    suffix = "_retraceability"

    cls_prefix = "RetraceExport"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        suffix,
        mocked_retraceability_export,
        xfail_prop="_expected_failure_retrace",
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
