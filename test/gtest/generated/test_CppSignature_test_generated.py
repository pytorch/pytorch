import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/CppSignature_test"


class TestCppSignatureTest(TestCase):
    cpp_name = "CppSignatureTest"

    def test_given_equalSignature_then_areEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "given_equalSignature_then_areEqual")

    def test_given_differentSignature_then_areDifferent(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "given_differentSignature_then_areDifferent"
        )

    def test_given_equalFunctorAndFunction_then_areEqual(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "given_equalFunctorAndFunction_then_areEqual"
        )

    def test_given_differentFunctorAndFunction_then_areDifferent(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "given_differentFunctorAndFunction_then_areDifferent",
        )

    def test_given_cppSignature_then_canQueryNameWithoutCrashing(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "given_cppSignature_then_canQueryNameWithoutCrashing",
        )


if __name__ == "__main__":
    run_tests()
