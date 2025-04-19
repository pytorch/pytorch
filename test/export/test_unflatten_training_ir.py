# Owner(s): ["oncall: export"]


try:
    from . import test_unflatten, testing
except ImportError:
    import test_unflatten  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch.export import export_for_training


test_classes = {}


def mocked_training_ir_export(*args, **kwargs):
    return export_for_training(*args, **kwargs)


def make_dynamic_cls(cls):
    cls_prefix = "TrainingIRUnflatten"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        "_training_ir",
        mocked_training_ir_export,
        xfail_prop="_expected_failure_training_ir",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__


tests = [
    test_unflatten.TestUnflatten,
]
for test in tests:
    make_dynamic_cls(test)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
