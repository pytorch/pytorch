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
    # Some test check for ending in suffix; need to make
    # the `_strict` for end of string as a result
    suffix = "_inline_and_install" + test_export.STRICT_SUFFIX

    cls_prefix = "InlineAndInstall"

    cls_a = testing.make_test_cls_with_mocked_export(
        cls,
        "StrictExport",
        suffix,
        mocked_strict_export,
        xfail_prop="_expected_failure_strict",
    )
    test_class = make_test_cls_with_patches(
        cls_a,
        cls_prefix,
        "",
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

## NOTE: The following tests are seemingly benign failures which are caused by string comparissons
## Fails because missing the `.slice()` in the string comparisson on the node
"""
AssertionError: String comparison failed: 'mod_list_1.2' != 'mod_list_1.slice(2, 3, None).2'
- mod_list_1.2
+ mod_list_1.slice(2, 3, None).2

"""
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_slice_nn_module_stack_inline_and_install_strict  # noqa: F821
)

## Fails because both point to the same installed module, which is `sub_net.0`
"""
  File "test/export/test_export.py", line 12441, in test_shared_submodule_nn_module_stack
    self.assertEqual(filtered_nn_module_stack[1], "sub_net.2")
AssertionError: String comparison failed: 'sub_net.0' != 'sub_net.2'
- sub_net.0
?         ^
+ sub_net.2
"""
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_shared_submodule_nn_module_stack_inline_and_install_strict  # noqa: F821
)

## Fails because installed name is different (has more __)
"""
AssertionError: String comparison failed: 'p_param_1' != 'p____parameters__param'
- p_param_1
+ p____parameters__param
"""
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_placeholder_naming_collisions_inline_and_install_strict  # noqa: F821
)

## Fails because of string reordering:
"""
- def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_weight, c_linear_bias, x, y):
?                                                                                                ---------------
+ def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_bias, c_linear_weight, x, y):
?                                                                              +++++++++++++++
"""
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_export_decomp_torture_case_2_inline_and_install_strict  # noqa: F821
)

## Fails because string reordering/renaming:
"""
- def forward(self, b_a_buffer, x):
+ def forward(self, b____modules__a____buffers__buffer, x):
      sym_size_int_1 = torch.ops.aten.sym_size.int(x, 0)
      gt = sym_size_int_1 > 4;  sym_size_int_1 = None
      true_graph_0 = self.true_graph_0
      false_graph_0 = self.false_graph_0
-     cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (x, b_a_buffer));  gt = true_graph_0 = false_graph_0 = x = b_a_buffer = None
+     cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (x, b____modules__a____buffers__buffer));  gt = true_graph_0 = false_graph_0 = x = b____modules__a____buffers__buffer = None
"""
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_export_cond_symbool_pred_inline_and_install_strict  # noqa: F821
)

# This test differs semantically from the original test, warrant further investigation

# NOTE: For this test, we have a failure that occurs because the buffers (for BatchNorm2D) are installed, and not
# graph input.  Therefore, they are not in the `program.graph_signature.inputs_to_buffers`
# and so not found by the unit test when counting the buffers
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_buffer_util_inline_and_install_strict  # noqa: F821
)

# NOTE: For this test, when we call `LOAD_ATTR`, we fail to realizing the LazyVariableTracker
# This is because the variable is popped off stack, pushed into TupleVariable (then ConstDictVariable)
# So, in the first case (not nested return), the LazyVariable is realized at the RETURN_VALUE call;
# for the second case (nested return), the LazyVariable is not realized until we begin COMPILING_GRAPH
# As a result, we don't install the variable, so crash when we expect the variable to be installed later
# Potential fix: We can force the lazy variable tracker to realize; just need to see how this is done for the non
# nested case
unittest.expectedFailure(
    InlineAndInstallStrictExportTestExport.test_constant_output_inline_and_install_strict  # noqa: F821
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
