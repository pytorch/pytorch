import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/half_float_ops_test"


class TestFloat16(TestCase):
    cpp_name = "Float16"

    def test_SimpleTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimpleTest")

    def test_UniformDistributionTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UniformDistributionTest")


if __name__ == "__main__":
    run_tests()
