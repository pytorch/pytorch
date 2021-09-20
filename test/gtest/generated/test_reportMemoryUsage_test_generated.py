import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/reportMemoryUsage_test"


class TestDefaultCPUAllocator(TestCase):
    cpp_name = "DefaultCPUAllocator"

    def test_check_reporter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "check_reporter")


if __name__ == "__main__":
    run_tests()
