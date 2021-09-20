import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/pow_test"


class TestPowTest(TestCase):
    cpp_name = "PowTest"

    def test_IntTensorPowAllScalars(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IntTensorPowAllScalars")

    def test_LongTensorPowAllScalars(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LongTensorPowAllScalars")

    def test_FloatTensorPowAllScalars(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FloatTensorPowAllScalars")

    def test_DoubleTensorPowAllScalars(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DoubleTensorPowAllScalars")

    def test_IntScalarPowAllTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IntScalarPowAllTensors")

    def test_LongScalarPowAllTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LongScalarPowAllTensors")

    def test_FloatScalarPowAllTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FloatScalarPowAllTensors")

    def test_DoubleScalarPowAllTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DoubleScalarPowAllTensors")

    def test_IntTensorPowIntTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IntTensorPowIntTensor")

    def test_LongTensorPowLongTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LongTensorPowLongTensor")

    def test_FloatTensorPowFloatTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FloatTensorPowFloatTensor")

    def test_DoubleTensorPowDoubleTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DoubleTensorPowDoubleTensor")

    def test_TestIntegralPow(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestIntegralPow")


if __name__ == "__main__":
    run_tests()
