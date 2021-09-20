import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/quantized_test"


class TestTestQTensor(TestCase):
    cpp_name = "TestQTensor"

    def test_QuantDequantAPIs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "QuantDequantAPIs")

    def test_RoundingMode(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RoundingMode")

    def test_Item(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Item")

    def test_EmptyQuantized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EmptyQuantized")

    def test_EmptyPerchannelQuantized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EmptyPerchannelQuantized")

    def test_QuantizePerChannel4d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "QuantizePerChannel4d")

    def test_QuantizePerChannel4dChannelsLast(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "QuantizePerChannel4dChannelsLast")

    def test_FromBlobQuantizedPerTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FromBlobQuantizedPerTensor")

    def test_FromBlobQuantizedPerChannel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FromBlobQuantizedPerChannel")


if __name__ == "__main__":
    run_tests()
