import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/scalar_tensor_test"


class TestTestScalarTensor(TestCase):
    cpp_name = "TestScalarTensor"

    def test_TestScalarTensorCPU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestScalarTensorCPU")

    def test_TestScalarTensorCUDA(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestScalarTensorCUDA")


if __name__ == "__main__":
    run_tests()
