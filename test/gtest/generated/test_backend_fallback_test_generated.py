import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/backend_fallback_test"


class TestBackendFallbackTest(TestCase):
    cpp_name = "BackendFallbackTest"

    def test_TestBackendFallbackWithMode(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBackendFallbackWithMode")

    def test_TestBackendFallbackWithWrapper(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBackendFallbackWithWrapper")

    def test_TestFallthroughBackendFallback(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestFallthroughBackendFallback")


if __name__ == "__main__":
    run_tests()
