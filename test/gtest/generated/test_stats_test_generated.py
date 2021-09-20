import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/stats_test"


class TestStatsTest(TestCase):
    cpp_name = "StatsTest"

    def test_StatsTestClass(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StatsTestClass")

    def test_StatsTestDuration(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StatsTestDuration")

    def test_StatsTestSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StatsTestSimple")

    def test_StatsTestStatic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StatsTestStatic")


if __name__ == "__main__":
    run_tests()
