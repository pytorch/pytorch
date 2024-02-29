# Owner(s): ["oncall: export"]

try:
    from . import test_export, testing
except ImportError:
    import test_export
    import testing
from torch.export import export
from torch._export.verifier import check_nn_module_stack

test_classes = {}

'''
Ideally this test file would not exist, and torch._export.verifier would call check_nn_module_stack() in its Verifier class.
For now nn_module_stack consistency is not fully covered (e.g. (de)serialization, export passes),
so this provides partial test coverage for the verifier.

TODO(pianpwk): move this to Verifier() once nn_module_stack consistency is fully covered.
'''

def mocked_export_check_nn_module_stack(*args, **kwargs):
    ep = export(*args, **kwargs)
    check_nn_module_stack(ep.graph_module)
    return ep


def make_dynamic_cls(cls):
    suffix = "_nn_module_stack"

    cls_prefix = "ExportCheckNNModuleStack"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        suffix,
        mocked_export_check_nn_module_stack,
        xfail_prop=None
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_export.TestExport
]
for test in tests:
    make_dynamic_cls(test)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
