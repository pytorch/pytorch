import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/converter_nomigraph_test"


class TestConverter(TestCase):
    cpp_name = "Converter"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_UnknownType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnknownType")

    def test_SpecializeConverter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SpecializeConverter")

    def test_ExternalInputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExternalInputs")

    def test_ExternalOutputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExternalOutputs")

    def test_InjectDataEdgeIndicators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InjectDataEdgeIndicators")


if __name__ == "__main__":
    run_tests()
