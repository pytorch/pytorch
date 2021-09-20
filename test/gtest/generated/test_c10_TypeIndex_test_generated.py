import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_TypeIndex_test"


class TestTypeIndex(TestCase):
    cpp_name = "TypeIndex"

    def test_TopLevelName(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TopLevelName")

    def test_NestedName(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NestedName")

    def test_TypeTemplateParameter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TypeTemplateParameter")

    def test_NonTypeTemplateParameter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NonTypeTemplateParameter")

    def test_TypeComputationsAreResolved(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TypeComputationsAreResolved")

    def test_FunctionTypeComputationsAreResolved(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FunctionTypeComputationsAreResolved")

    def test_FunctionArgumentsAndReturns(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FunctionArgumentsAndReturns")


if __name__ == "__main__":
    run_tests()
