import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/test_dist_autograd"


class TestDistAutogradTest(TestCase):
    cpp_name = "DistAutogradTest"

    def test_TestSendFunctionInvalidInputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSendFunctionInvalidInputs")

    def test_TestInitializedContextCleanup(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestInitializedContextCleanup")

    def test_TestInitializedContextCleanupSendFunction(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestInitializedContextCleanupSendFunction"
        )


if __name__ == "__main__":
    run_tests()
