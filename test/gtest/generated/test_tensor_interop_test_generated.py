import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/tensor_interop_test"


class TestCaffe2ToPytorch(TestCase):
    cpp_name = "Caffe2ToPytorch"

    def test_SimpleLegacy(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimpleLegacy")

    def test_Simple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Simple")

    def test_ExternalData(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExternalData")

    def test_Op(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Op")

    def test_PartiallyInitialized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PartiallyInitialized")

    def test_MutualResizes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MutualResizes")

    def test_NonPOD(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NonPOD")

    def test_Nullptr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Nullptr")


class TestPytorchToCaffe2(TestCase):
    cpp_name = "PytorchToCaffe2"

    def test_Op(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Op")

    def test_SharedStorageRead(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SharedStorageRead")

    def test_SharedStorageWrite(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SharedStorageWrite")

    def test_MutualResizes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MutualResizes")

    def test_Strided(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Strided")

    def test_InplaceStrided(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InplaceStrided")

    def test_NonRegularTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NonRegularTensor")

    def test_Nullptr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Nullptr")


if __name__ == "__main__":
    run_tests()
