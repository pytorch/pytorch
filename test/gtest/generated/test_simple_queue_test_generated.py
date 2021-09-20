import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/simple_queue_test"


class TestSimpleQueueDeathTest(TestCase):
    cpp_name = "SimpleQueueDeathTest"

    def test_CannotAddAfterQueueFinished(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CannotAddAfterQueueFinished")


class TestSimpleQueueTest(TestCase):
    cpp_name = "SimpleQueueTest"

    def test_SingleProducerSingleConsumer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SingleProducerSingleConsumer")

    def test_SingleProducerDoubleConsumer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SingleProducerDoubleConsumer")

    def test_DoubleProducerDoubleConsumer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DoubleProducerDoubleConsumer")


if __name__ == "__main__":
    run_tests()
