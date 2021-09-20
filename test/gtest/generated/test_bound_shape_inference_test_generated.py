import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/bound_shape_inference_test"


class TestBoundShapeInference(TestCase):
    cpp_name = "BoundShapeInference"

    def test_SparseLengthsSum(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SparseLengthsSum")

    def test_SparseLengthsSumSparseLookup(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SparseLengthsSumSparseLookup")

    def test_SparseLengthsSumFused8BitRowwise(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SparseLengthsSumFused8BitRowwise")

    def test_SparseLengthsSum8BitRowwiseSparse(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SparseLengthsSum8BitRowwiseSparse")

    def test_SparseLengthsSumFused4BitRowwise(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SparseLengthsSumFused4BitRowwise")

    def test_LengthsRangeFill(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LengthsRangeFill")

    def test_ConstantFill(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantFill")

    def test_Reshape(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Reshape")

    def test_ConcatMissingInput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConcatMissingInput")

    def test_Int8QuantizeInferInputBackwards(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Int8QuantizeInferInputBackwards")

    def test_ConcatInferInputBackwards(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConcatInferInputBackwards")

    def test_ElementwiseInferInputBackwards(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ElementwiseInferInputBackwards")

    def test_ElementwiseOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ElementwiseOp")

    def test_Bucketize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Bucketize")

    def test_Split(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Split")

    def test_FC(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FC")

    def test_FC3D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FC3D")

    def test_Quantization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Quantization")

    def test_Tile(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Tile")

    def test_Combo0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Combo0")

    def test_Softmax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softmax")

    def test_LpNorm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LpNorm")

    def test_Transpose(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Transpose")


if __name__ == "__main__":
    run_tests()
