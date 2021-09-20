import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/vmap_test"


class TestVmapTest(TestCase):
    cpp_name = "VmapTest"

    def test_TestBatchedTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBatchedTensor")

    def test_TestBatchedTensorMaxLevel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBatchedTensorMaxLevel")

    def test_TestBatchedTensorActualDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBatchedTensorActualDim")

    def test_TestMultiBatchVmapTransform(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestMultiBatchVmapTransform")

    def test_TestVmapPhysicalViewGetPhysicalDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestVmapPhysicalViewGetPhysicalDim")

    def test_TestVmapPhysicalViewGetPhysicalDims(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestVmapPhysicalViewGetPhysicalDims")

    def test_TestVmapPhysicalViewNewLogicalFromPhysical(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestVmapPhysicalViewNewLogicalFromPhysical"
        )

    def test_TestBatchedTensorSum(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBatchedTensorSum")

    def test_TestBroadcastingVmapTransformBatchedBatched(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestBroadcastingVmapTransformBatchedBatched"
        )

    def test_TestBroadcastingVmapTransformBatchedUnbatched(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestBroadcastingVmapTransformBatchedUnbatched"
        )

    def test_TestBroadcastingVmapTransformMaxLevels(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestBroadcastingVmapTransformMaxLevels"
        )

    def test_TestBatchedTensorMul(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBatchedTensorMul")

    def test_TestBatchedTensorSize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBatchedTensorSize")

    def test_TestVmapPhysicalViewGetPhysicalShape(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestVmapPhysicalViewGetPhysicalShape")

    def test_TestBatchedTensorExpand(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBatchedTensorExpand")

    def test_TestBatchedTensorUnsqueeze(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBatchedTensorUnsqueeze")

    def test_TestBatchedTensorSqueeze(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBatchedTensorSqueeze")

    def test_TestBatchedTensorTranspose(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBatchedTensorTranspose")

    def test_TestBatchedTensorPermute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBatchedTensorPermute")

    def test_TestMultiBatchVmapTransformBatchedBatched(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestMultiBatchVmapTransformBatchedBatched"
        )

    def test_TestMultiBatchVmapTransformBatchedUnbatched(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestMultiBatchVmapTransformBatchedUnbatched"
        )

    def test_TestMultiBatchVmapTransformMaxLevels(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestMultiBatchVmapTransformMaxLevels")

    def test_TestMultiBatchVmapTransformMultipleTensors(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestMultiBatchVmapTransformMultipleTensors"
        )


if __name__ == "__main__":
    run_tests()
