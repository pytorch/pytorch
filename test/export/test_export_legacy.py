# Owner(s): ["oncall: export"]
import torch


try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library

    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch.testing._internal.common_utils import IS_FBCODE


if IS_FBCODE:
    from pyjk import PyPatchJustKnobs


test_classes = {}


def mocked_legacy_export(*args, **kwargs):
    with PyPatchJustKnobs().patch(
        "pytorch/export:export_training_ir_rollout_check", False
    ):
        return torch.export._trace._export(*args, **kwargs, pre_dispatch=True)


def mocked_legacy_export_non_strict(*args, **kwargs):
    with PyPatchJustKnobs().patch(
        "pytorch/export:export_training_ir_rollout_check", False
    ):
        if "strict" in kwargs:
            return torch.export._trace._export(*args, **kwargs, pre_dispatch=True)
        return torch.export._trace._export(
            *args, **kwargs, pre_dispatch=True, strict=False
        )


def make_dynamic_cls(cls, strict):
    if strict:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "LegacyExport",
            test_export.LEGACY_EXPORT_STRICT_SUFFIX,
            mocked_legacy_export,
            xfail_prop="_expected_failure_legacy_export",
        )
    else:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "LegacyExportNonStrict",
            test_export.LEGACY_EXPORT_NONSTRICT_SUFFIX,
            mocked_legacy_export_non_strict,
            xfail_prop="_expected_failure_legacy_export_non_strict",
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

if IS_FBCODE:
    for test in tests:
        make_dynamic_cls(test, True)
        make_dynamic_cls(test, False)
    del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if IS_FBCODE:
        run_tests()
