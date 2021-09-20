import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/elementwise_op_test"


class TestElementwiseCPUTest(TestCase):
    cpp_name = "ElementwiseCPUTest"

    def test_And(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "And")


class TestElementwiseTest(TestCase):
    cpp_name = "ElementwiseTest"

    def test_Or(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Or")

    def test_Xor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Xor")

    def test_Not(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Not")

    def test_EQ(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EQ")


if __name__ == "__main__":
    run_tests()
