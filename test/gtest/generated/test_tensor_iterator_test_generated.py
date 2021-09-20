import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/tensor_iterator_test"


class TestTensorIteratorTest(TestCase):
    cpp_name = "TensorIteratorTest"

    def test_CPUScalar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CPUScalar")

    def test_CPUScalarInputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CPUScalarInputs")

    def test_MixedDevices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MixedDevices")

    def test_SerialLoopUnary_Byte(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnary_Byte")

    def test_SerialLoopUnary_Char(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnary_Char")

    def test_SerialLoopUnary_Short(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnary_Short")

    def test_SerialLoopUnary_Int(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnary_Int")

    def test_SerialLoopUnary_Long(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnary_Long")

    def test_SerialLoopUnary_Float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnary_Float")

    def test_SerialLoopUnary_Double(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnary_Double")

    def test_SerialLoopBinary_Byte(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinary_Byte")

    def test_SerialLoopBinary_Char(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinary_Char")

    def test_SerialLoopBinary_Short(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinary_Short")

    def test_SerialLoopBinary_Int(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinary_Int")

    def test_SerialLoopBinary_Long(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinary_Long")

    def test_SerialLoopBinary_Float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinary_Float")

    def test_SerialLoopBinary_Double(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinary_Double")

    def test_SerialLoopPointwise_Byte(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPointwise_Byte")

    def test_SerialLoopPointwise_Char(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPointwise_Char")

    def test_SerialLoopPointwise_Short(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPointwise_Short")

    def test_SerialLoopPointwise_Int(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPointwise_Int")

    def test_SerialLoopPointwise_Long(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPointwise_Long")

    def test_SerialLoopPointwise_Float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPointwise_Float")

    def test_SerialLoopPointwise_Double(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPointwise_Double")

    def test_SerialLoopUnaryNoOutput_Byte(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnaryNoOutput_Byte")

    def test_SerialLoopUnaryNoOutput_Char(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnaryNoOutput_Char")

    def test_SerialLoopUnaryNoOutput_Short(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnaryNoOutput_Short")

    def test_SerialLoopUnaryNoOutput_Int(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnaryNoOutput_Int")

    def test_SerialLoopUnaryNoOutput_Long(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnaryNoOutput_Long")

    def test_SerialLoopUnaryNoOutput_Float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnaryNoOutput_Float")

    def test_SerialLoopUnaryNoOutput_Double(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopUnaryNoOutput_Double")

    def test_SerialLoopBinaryNoOutput_Byte(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinaryNoOutput_Byte")

    def test_SerialLoopBinaryNoOutput_Char(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinaryNoOutput_Char")

    def test_SerialLoopBinaryNoOutput_Short(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinaryNoOutput_Short")

    def test_SerialLoopBinaryNoOutput_Int(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinaryNoOutput_Int")

    def test_SerialLoopBinaryNoOutput_Long(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinaryNoOutput_Long")

    def test_SerialLoopBinaryNoOutput_Float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinaryNoOutput_Float")

    def test_SerialLoopBinaryNoOutput_Double(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopBinaryNoOutput_Double")

    def test_SerialLoopPoinwiseNoOutput_Byte(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPoinwiseNoOutput_Byte")

    def test_SerialLoopPoinwiseNoOutput_Char(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPoinwiseNoOutput_Char")

    def test_SerialLoopPoinwiseNoOutput_Short(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPoinwiseNoOutput_Short")

    def test_SerialLoopPoinwiseNoOutput_Int(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPoinwiseNoOutput_Int")

    def test_SerialLoopPoinwiseNoOutput_Long(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPoinwiseNoOutput_Long")

    def test_SerialLoopPoinwiseNoOutput_Float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPoinwiseNoOutput_Float")

    def test_SerialLoopPoinwiseNoOutput_Double(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopPoinwiseNoOutput_Double")

    def test_ComparisonLoopBinary_Byte(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ComparisonLoopBinary_Byte")

    def test_ComparisonLoopBinary_Char(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ComparisonLoopBinary_Char")

    def test_ComparisonLoopBinary_Short(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ComparisonLoopBinary_Short")

    def test_ComparisonLoopBinary_Int(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ComparisonLoopBinary_Int")

    def test_ComparisonLoopBinary_Long(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ComparisonLoopBinary_Long")

    def test_ComparisonLoopBinary_Float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ComparisonLoopBinary_Float")

    def test_ComparisonLoopBinary_Double(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ComparisonLoopBinary_Double")

    def test_ComparisonLoopBinary_Bool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ComparisonLoopBinary_Bool")

    def test_SerialLoopSingleThread(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SerialLoopSingleThread")

    def test_InputDType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InputDType")

    def test_ComputeCommonDTypeInputOnly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ComputeCommonDTypeInputOnly")

    def test_DoNotComputeCommonDTypeInputOnly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DoNotComputeCommonDTypeInputOnly")

    def test_FailNonPromotingBinaryOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FailNonPromotingBinaryOp")

    def test_CpuKernelMultipleOutputs_Byte(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CpuKernelMultipleOutputs_Byte")

    def test_CpuKernelMultipleOutputs_Char(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CpuKernelMultipleOutputs_Char")

    def test_CpuKernelMultipleOutputs_Short(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CpuKernelMultipleOutputs_Short")

    def test_CpuKernelMultipleOutputs_Int(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CpuKernelMultipleOutputs_Int")

    def test_CpuKernelMultipleOutputs_Long(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CpuKernelMultipleOutputs_Long")

    def test_CpuKernelMultipleOutputs_Float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CpuKernelMultipleOutputs_Float")

    def test_CpuKernelMultipleOutputs_Double(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CpuKernelMultipleOutputs_Double")


if __name__ == "__main__":
    run_tests()
