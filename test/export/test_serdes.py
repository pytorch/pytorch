# Owner(s): ["oncall: export"]

import io

try:
    from . import test_export, testing
except ImportError:
    import test_export
    import testing

from torch.export import export, load, save
from torch.export._trace import _export

test_classes = {}


def mocked_serder_export(*args, **kwargs):
    ep = export(*args, **kwargs)
    buffer = io.BytesIO()
    save(ep, buffer)
    buffer.seek(0)
    loaded_ep = load(buffer)
    return loaded_ep


def mocked_serder_export_pre_dispatch(*args, **kwargs):
    ep = _export(*args, **kwargs, pre_dispatch=True)
    buffer = io.BytesIO()
    save(ep, buffer)
    buffer.seek(0)
    loaded_ep = load(buffer)
    return loaded_ep


def make_dynamic_cls(cls):
    suffix = "_serdes"
    suffix_pre_dispatch = "_serdes_pre_dispatch"

    cls_prefix = "SerDesExport"
    cls_prefix_pre_dispatch = "SerDesExportPreDispatch"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        suffix,
        mocked_serder_export,
        xfail_prop="_expected_failure_serdes",
    )

    test_class_pre_dispatch = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix_pre_dispatch,
        suffix_pre_dispatch,
        mocked_serder_export_pre_dispatch,
        xfail_prop="_expected_failure_serdes_pre_dispatch",
    )

    test_classes[test_class.__name__] = test_class
    test_classes[test_class_pre_dispatch.__name__] = test_class_pre_dispatch
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    globals()[test_class_pre_dispatch.__name__] = test_class_pre_dispatch
    test_class.__module__ = __name__
    test_class_pre_dispatch.__module__ = __name__


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
