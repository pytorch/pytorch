import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/native_test"


class TestTestNative(TestCase):
    cpp_name = "TestNative"

    def test_NativeTestCPU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NativeTestCPU")

    def test_NativeTestGPU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NativeTestGPU")


if __name__ == "__main__":
    run_tests()
