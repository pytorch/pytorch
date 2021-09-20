import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_DispatchKeySet_test"


class TestDispatchKeySet(TestCase):
    cpp_name = "DispatchKeySet"

    def test_Empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Empty")

    def test_Singleton(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Singleton")

    def test_Doubleton(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Doubleton")

    def test_Full(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Full")

    def test_IteratorBasicOps(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IteratorBasicOps")

    def test_IteratorEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IteratorEmpty")

    def test_IteratorFull(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IteratorFull")

    def test_IteratorRangeFull(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IteratorRangeFull")

    def test_SpecificKeys(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SpecificKeys")

    def test_FailAtEndIterator(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FailAtEndIterator")


if __name__ == "__main__":
    run_tests()
