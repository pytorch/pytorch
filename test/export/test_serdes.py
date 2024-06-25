# Owner(s): ["oncall: export"]

import io

try:
    from . import test_export, testing
except ImportError:
    import test_export
    import testing

from torch.export import export, load, save

test_classes = {}


def mocked_serder_export(*args, **kwargs):
    ep = export(*args, **kwargs)
    buffer = io.BytesIO()
    save(ep, buffer)
    buffer.seek(0)
    loaded_ep = load(buffer)
    return loaded_ep


def make_dynamic_cls(cls):
    cls_prefix = "SerDesExport"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        test_export.SERDES_SUFFIX,
        mocked_serder_export,
        xfail_prop="_expected_failure_serdes",
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
    make_dynamic_cls(test)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
