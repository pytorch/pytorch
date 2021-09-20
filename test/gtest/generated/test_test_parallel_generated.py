import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/test_parallel"


class TestTestParallel(TestCase):
    cpp_name = "TestParallel"

    def test_TestParallel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestParallel")

    def test_NestedParallel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NestedParallel")

    def test_NestedParallelThreadId(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NestedParallelThreadId")

    def test_Exceptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Exceptions")

    def test_IntraOpLaunchFuture(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IntraOpLaunchFuture")


if __name__ == "__main__":
    run_tests()
