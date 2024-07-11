# Owner(s): ["oncall: export"]

try:
    from . import test_export, testing
except ImportError:
    import test_export
    import testing
from torch.export._trace import _export_for_training

test_classes = {}


def mocked_training_ir_to_run_decomp_export_non_strict(*args, **kwargs):
    if "strict":
        ep = _export_for_training(*args, **kwargs)
    else:
        ep = _export_for_training(*args, **kwargs, strict=False)
    return ep.run_decompositions(
        {}, _preserve_ops=testing._COMPOSITE_OPS_THAT_CAN_BE_PRESERVED_TESTING_ONLY
    )


def make_dynamic_cls(cls):
    cls_prefix_non_strict = "TrainingIRToRunDecompExportNonStrict"

    test_class_non_strict = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix_non_strict,
        test_export.TRAINING_IR_DECOMP_NS_SUFFIX,
        mocked_training_ir_to_run_decomp_export_non_strict,
        xfail_prop="_expected_failure_training_ir_to_run_decomp_non_strict",
    )

    test_classes[test_class_non_strict.__name__] = test_class_non_strict
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class_non_strict.__name__] = test_class_non_strict
    test_class_non_strict.__module__ = __name__
    return test_class_non_strict


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
