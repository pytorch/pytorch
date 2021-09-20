import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/graph_test"


class TestGraphTest(TestCase):
    cpp_name = "GraphTest"

    def test_TestGenerateGraphChain(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestGenerateGraphChain")

    def test_TestGenerateGraphChainInPlace(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestGenerateGraphChainInPlace")

    def test_TestGenerateGraphBranch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestGenerateGraphBranch")

    def test_TestReusedInputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestReusedInputs")

    def test_TestGetPerimeter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestGetPerimeter")


if __name__ == "__main__":
    run_tests()
