import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/event_test"


class TestEventCPUTest(TestCase):
    cpp_name = "EventCPUTest"

    def test_EventBasics(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EventBasics")

    def test_EventErrors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EventErrors")


if __name__ == "__main__":
    run_tests()
