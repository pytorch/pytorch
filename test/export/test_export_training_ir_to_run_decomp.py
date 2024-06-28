# Owner(s): ["oncall: export"]

try:
    from . import test_export, testing
except ImportError:
    import test_export
    import testing
from torch.export._trace import _export_for_training

test_classes = {}


def mocked_training_ir_to_run_decomp_export(*args, **kwargs):
    ep = _export_for_training(*args, **kwargs)
    return ep.run_decompositions(
        {}, _preserve_ops=testing._COMPOSITE_OPS_THAT_CAN_BE_PRESERVED_TESTING_ONLY
    )


def make_dynamic_cls(cls):
    cls_prefix = "TrainingIRToRunDecompExport"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        test_export.TRAINING_IR_DECOMP_SUFFIX,
        mocked_training_ir_to_run_decomp_export,
        xfail_prop="_expected_failure_training_ir_to_run_decomp",
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
