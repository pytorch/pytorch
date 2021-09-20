import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/memory_format_test"


class TestMemoryFormatTest(TestCase):
    cpp_name = "MemoryFormatTest"

    def test_SetMemoryFormat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetMemoryFormat")

    def test_TransposeMemoryFormat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TransposeMemoryFormat")

    def test_SliceStepTwoMemoryFormat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SliceStepTwoMemoryFormat")

    def test_SliceFirstMemoryFormat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SliceFirstMemoryFormat")


if __name__ == "__main__":
    run_tests()
