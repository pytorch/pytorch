import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_logging_test"


class TestLoggingDeathTest(TestCase):
    cpp_name = "LoggingDeathTest"

    def test_TestEnforceUsingFatal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEnforceUsingFatal")


class TestLoggingTest(TestCase):
    cpp_name = "LoggingTest"

    def test_TestEnforceTrue(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEnforceTrue")

    def test_TestEnforceFalse(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEnforceFalse")

    def test_TestEnforceEquals(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEnforceEquals")

    def test_TestEnforceMessageVariables(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEnforceMessageVariables")

    def test_EnforceEqualsObjectWithReferenceToTemporaryWithoutUseOutOfScope(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "EnforceEqualsObjectWithReferenceToTemporaryWithoutUseOutOfScope",
        )

    def test_DoesntCopyComparedObjects(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DoesntCopyComparedObjects")

    def test_EnforceShowcase(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EnforceShowcase")

    def test_Join(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Join")

    def test_TestDanglingElse(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDanglingElse")


if __name__ == "__main__":
    run_tests()
