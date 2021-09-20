import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/nnpack_test"


class TestNNPACK(TestCase):
    cpp_name = "NNPACK"

    def test_Conv_3x3s1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv_3x3s1")

    def test_Conv_3x3s1_precompute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv_3x3s1_precompute")

    def test_Conv_3x3s1_FP16(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv_3x3s1_FP16")

    def test_Conv_3x3s1_FP16_precompute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv_3x3s1_FP16_precompute")

    def test_Conv_NxNs1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv_NxNs1")

    def test_Conv_1x1s1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv_1x1s1")

    def test_ConvRelu_1x1s1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvRelu_1x1s1")

    def test_Conv_1x1s1_precompute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv_1x1s1_precompute")

    def test_Conv_NxNs_grouped(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv_NxNs_grouped")

    def test_Conv_NxNs_grouped_precompute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv_NxNs_grouped_precompute")

    def test_Conv_NxNsW(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv_NxNsW")

    def test_ConvRelu_NxNsW(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvRelu_NxNsW")

    def test_Conv_HxWsHxW(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv_HxWsHxW")


if __name__ == "__main__":
    run_tests()
