import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/int8_test"


class TestInt8(TestCase):
    cpp_name = "Int8"

    def test_ReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReLU")

    def test_DISABLED_LeakyReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_LeakyReLU")

    def test_Softmax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softmax")

    def test_Sigmoid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Sigmoid")

    def test_MaxPool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool")

    def test_AveragePool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AveragePool")

    def test_ResizeNearest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ResizeNearest")

    def test_ChannelShuffle(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ChannelShuffle")

    def test_Concat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Concat")

    def test_Add(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Add")

    def test_SumRelu(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SumRelu")

    def test_Conv(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv")

    def test_Grouped1x1Conv(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Grouped1x1Conv")

    def test_Conv2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv2")

    def test_DepthwiseConv(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DepthwiseConv")

    def test_DepthwiseConv3x3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DepthwiseConv3x3")

    def test_DepthwiseConv5x5(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DepthwiseConv5x5")

    def test_ConvTranspose(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvTranspose")

    def test_FC(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FC")

    def test_GivenTensorFill(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GivenTensorFill")

    def test_GivenIntTensorFill(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GivenIntTensorFill")

    def test_QuantDeQuant(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "QuantDeQuant")

    def test_Reshape(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Reshape")

    def test_Flatten(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Flatten")

    def test_Slice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Slice")

    def test_DISABLED_Transpose(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_Transpose")


if __name__ == "__main__":
    run_tests()
