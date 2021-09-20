import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/operator_schema_test"


class TestOperatorSchemaTest(TestCase):
    cpp_name = "OperatorSchemaTest"

    def test_BasicSchema(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BasicSchema")

    def test_SpecifiedInputOutput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SpecifiedInputOutput")

    def test_InputOutputRelation(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InputOutputRelation")

    def test_SameInputOutput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SameInputOutput")

    def test_CalculateOutput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CalculateOutput")

    def test_Inplace(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Inplace")

    def test_TensorInferenceIdentical(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInferenceIdentical")

    def test_TensorInferenceArbitrary(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInferenceArbitrary")

    def test_TestCastSchema(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestCastSchema")

    def test_TestCostInference(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestCostInference")


if __name__ == "__main__":
    run_tests()
