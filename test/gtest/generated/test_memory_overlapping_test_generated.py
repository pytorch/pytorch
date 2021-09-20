import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/memory_overlapping_test"


class TestMemoryOverlapTest(TestCase):
    cpp_name = "MemoryOverlapTest"

    def test_TensorExpanded(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorExpanded")

    def test_ScalarExpanded(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScalarExpanded")

    def test_NonContiguousTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NonContiguousTensor")

    def test_NonContiguousExpandedTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NonContiguousExpandedTensor")

    def test_ContiguousTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ContiguousTensor")

    def test_ContiguousExpandedTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ContiguousExpandedTensor")


if __name__ == "__main__":
    run_tests()
