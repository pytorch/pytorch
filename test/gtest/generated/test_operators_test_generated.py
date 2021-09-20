import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/operators_test"


class TestOperatorsTest(TestCase):
    cpp_name = "OperatorsTest"

    def test_TestFunctionDecltype(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestFunctionDecltype")

    def test_TestMethodOnlyDecltype(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestMethodOnlyDecltype")

    def test_Test_ATEN_FN(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Test_ATEN_FN")

    def test_TestOutVariantIsFaithful(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestOutVariantIsFaithful")


if __name__ == "__main__":
    run_tests()
