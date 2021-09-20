import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/cpu_profiling_allocator_test"


class TestCPUAllocationPlanTest(TestCase):
    cpp_name = "CPUAllocationPlanTest"

    def test_with_control_flow(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "with_control_flow")

    def test_with_profiling_alloc(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "with_profiling_alloc")


if __name__ == "__main__":
    run_tests()
