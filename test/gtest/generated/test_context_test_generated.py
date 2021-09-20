import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/context_test"


class TestCPUContextTest(TestCase):
    cpp_name = "CPUContextTest"

    def test_TestAllocAlignment(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestAllocAlignment")

    def test_TestAllocDealloc(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestAllocDealloc")


if __name__ == "__main__":
    run_tests()
