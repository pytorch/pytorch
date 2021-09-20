import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/workspace_test"


class TestWorkspaceTest(TestCase):
    cpp_name = "WorkspaceTest"

    def test_BlobAccess(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BlobAccess")

    def test_RunEmptyPlan(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RunEmptyPlan")

    def test_Sharing(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Sharing")

    def test_BlobMapping(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BlobMapping")

    def test_ForEach(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ForEach")


if __name__ == "__main__":
    run_tests()
