import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/init_test"


class TestInitTest(TestCase):
    cpp_name = "InitTest"

    def test_TestInitFunctionHasRun(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestInitFunctionHasRun")

    def test_CanRerunGlobalInit(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanRerunGlobalInit")

    def test_FailLateRegisterInitFunction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FailLateRegisterInitFunction")


if __name__ == "__main__":
    run_tests()
