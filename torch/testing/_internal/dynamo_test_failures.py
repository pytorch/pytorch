# We generate unittest.expectedFailure for all of the following tests
# when run under PYTORCH_TEST_WITH_DYNAMO=1.
#
# This lists exists so we can more easily add large numbers of failing tests,
dynamo_expected_failures = {
    "TestCppExtensionJIT.test_cpp_frontend_module_has_up_to_date_attribute",
    "TestCppExtensionJIT.test_custom_compound_op_autograd",
    "TestCppExtensionJIT.test_cpp_frontend_module_has_up_to_date_attributes",
    "TestCppExtensionOpenRgistration.test_open_device_registration",
}
