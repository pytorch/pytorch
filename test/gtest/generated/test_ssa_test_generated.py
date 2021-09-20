import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/ssa_test"


class TestSsaTest(TestCase):
    cpp_name = "SsaTest"

    def test_ConvReluInplace(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvReluInplace")

    def test_FC_Relu_FC_InPlace_Output(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FC_Relu_FC_InPlace_Output")


if __name__ == "__main__":
    run_tests()
