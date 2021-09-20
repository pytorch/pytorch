import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/scalar_test"


class TestTestScalar(TestCase):
    cpp_name = "TestScalar"

    def test_TestScalar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestScalar")

    def test_TestConj(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestConj")

    def test_TestEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEqual")

    def test_TestFormatting(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestFormatting")


if __name__ == "__main__":
    run_tests()
