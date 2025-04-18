# Owner(s): ["oncall: export"]


import unittest

from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches


try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch.export import export


test_classes = {}


def mocked_strict_export(*args, **kwargs):
    # If user already specified strict, don't make it strict
    if "strict" in kwargs:
        return export(*args, **kwargs)
    return export(*args, **kwargs, strict=True)


def make_dynamic_cls(cls):
    suffix = "_inline_and_install"

    cls_prefix = "InlineAndInstall"

    cls_a = testing.make_test_cls_with_mocked_export(
        cls,
        "StrictExport",
        test_export.STRICT_SUFFIX,
        mocked_strict_export,
        xfail_prop="_expected_failure_strict",
    )
    test_class = make_test_cls_with_patches(
        cls_a,
        cls_prefix,
        suffix,
        (config, "install_params_as_graph_attr", True),
        (config, "inline_inbuilt_nn_modules", True),
        xfail_prop="_expected_failure_inline_and_install",
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


# After installing and inlining is turned on, these tests won't throw
# errors in export (which is expected for the test to pass)
# Therefore, these unittest are expected to fail, and we need to update the
# semantics
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_real_tensor_for_max_op_strict_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_unflatten_multiple_graphs_shared_submodule_strict_inline_and_install  # noqa: F821
)


# These test check for some string comparisson that now fails, perhaps due to different naming or ordering
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_slice_nn_module_stack_strict_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_shared_submodule_nn_module_stack_strict_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_placeholder_naming_collisions_strict_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_export_decomp_torture_case_2_strict_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_export_cond_symbool_pred_strict_inline_and_install  # noqa: F821
)

# This test differs semantically from the original test, warrant further investigation
# TODO:[lucaskabela] Debug these tests more in depth before turning on for export
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_buffer_util_strict_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_real_tensor_size_mismatch_strict_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_real_tensor_alias_dtype_mismatch_strict_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_multidimensional_slicing_strict_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_error_does_not_reference_eager_fallback_strict_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_constant_output_strict_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_check_is_size_error_strict_inline_and_install  # noqa: F821
)
unittest.expectedFailure(
    InlineAndInstallStrictExportTestDynamismExpression.test_export_constraints_error_not_in_range_strict_inline_and_install  # noqa: F821
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
