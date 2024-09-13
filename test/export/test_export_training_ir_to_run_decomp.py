# Owner(s): ["oncall: export"]
import torch


try:
    from . import test_export, testing
except ImportError:
    import test_export

    import testing


test_classes = {}


def mocked_training_ir_to_run_decomp_export_strict(*args, **kwargs):
    ep = torch.export.export_for_training(*args, **kwargs)
    return ep.run_decompositions(
        {}, _preserve_ops=testing._COMPOSITE_OPS_THAT_CAN_BE_PRESERVED_TESTING_ONLY
    )


def mocked_training_ir_to_run_decomp_export_non_strict(*args, **kwargs):
    if "strict" in kwargs:
        ep = torch.export.export_for_training(*args, **kwargs)
    else:
        ep = torch.export.export_for_training(*args, **kwargs, strict=False)
    return ep.run_decompositions(
        {}, _preserve_ops=testing._COMPOSITE_OPS_THAT_CAN_BE_PRESERVED_TESTING_ONLY
    )


def make_dynamic_cls(cls, strict):
    if strict:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "TrainingIRToRunDecompExport",
            test_export.TRAINING_IR_DECOMP_STRICT_SUFFIX,
            mocked_training_ir_to_run_decomp_export_strict,
            xfail_prop="_expected_failure_training_ir_to_run_decomp",
        )
    else:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "TrainingIRToRunDecompExportNonStrict",
            test_export.TRAINING_IR_DECOMP_NON_STRICT_SUFFIX,
            mocked_training_ir_to_run_decomp_export_non_strict,
            xfail_prop="_expected_failure_training_ir_to_run_decomp_non_strict",
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
