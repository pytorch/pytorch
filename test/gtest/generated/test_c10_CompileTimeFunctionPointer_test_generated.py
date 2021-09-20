import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_CompileTimeFunctionPointer_test"


class TestCompileTimeFunctionPointerTest(TestCase):
    cpp_name = "CompileTimeFunctionPointerTest"

    def test_runFunctionThroughType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "runFunctionThroughType")

    def test_runFunctionThroughValue(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "runFunctionThroughValue")


if __name__ == "__main__":
    run_tests()
