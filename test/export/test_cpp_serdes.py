# Owner(s): ["oncall: export"]


import torch
from torch._export.serde.serialize import deserialize, serialize


try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch.export import export


test_classes = {}


def mocked_cpp_serdes_export(*args, **kwargs):
    ep = export(*args, **kwargs)
    try:
        payload = serialize(ep)
    except Exception:
        return ep
    cpp_ep = torch._C._export.deserialize_exported_program(payload.exported_program)
    loaded_json = torch._C._export.serialize_exported_program(cpp_ep)
    payload.exported_program = loaded_json.encode()
    loaded_ep = deserialize(payload)
    return loaded_ep


def make_dynamic_cls(cls):
    cls_prefix = "CppSerdes"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        "_cpp_serdes",
        mocked_cpp_serdes_export,
        xfail_prop="_expected_failure_cpp_serdes",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__


tests = [
    test_export.TestExport,
]
for test in tests:
    make_dynamic_cls(test)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
