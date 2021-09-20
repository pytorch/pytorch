import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/cpuid_test"


class TestCpuIdTest(TestCase):
    cpp_name = "CpuIdTest"

    def test_ShouldAlwaysHaveMMX(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ShouldAlwaysHaveMMX")


if __name__ == "__main__":
    run_tests()
