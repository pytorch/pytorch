import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/net_async_tracing_test"


class TestNetAsyncTracingTest(TestCase):
    cpp_name = "NetAsyncTracingTest"

    def test_ExtractShardId(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExtractShardId")

    def test_EveryKIteration(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EveryKIteration")

    def test_GlobalTimeSlice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GlobalTimeSlice")


if __name__ == "__main__":
    run_tests()
