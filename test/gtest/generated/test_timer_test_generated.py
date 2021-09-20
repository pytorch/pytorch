import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/timer_test"


class TestTimerTest(TestCase):
    cpp_name = "TimerTest"

    def test_Test(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Test")

    def test_TestLatency(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestLatency")


if __name__ == "__main__":
    run_tests()
