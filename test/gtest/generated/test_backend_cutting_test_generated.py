import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/backend_cutting_test"


class TestBackendCuttingTest(TestCase):
    cpp_name = "BackendCuttingTest"

    def test_unit(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "unit")

    def test_line(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "line")

    def test_convergedPaths(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "convergedPaths")

    def test_skipPath(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "skipPath")


if __name__ == "__main__":
    run_tests()
