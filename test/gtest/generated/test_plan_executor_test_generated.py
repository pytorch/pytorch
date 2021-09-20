import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/plan_executor_test"


class TestPlanExecutorTest(TestCase):
    cpp_name = "PlanExecutorTest"

    def test_EmptyPlan(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EmptyPlan")

    def test_ErrorAsyncPlan(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ErrorAsyncPlan")

    def test_BlockingErrorPlan(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BlockingErrorPlan")

    def test_ErrorPlanWithCancellableStuckNet(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ErrorPlanWithCancellableStuckNet")

    def test_ReporterErrorPlanWithCancellableStuckNet(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ReporterErrorPlanWithCancellableStuckNet"
        )

    def test_ShouldStopWithCancel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ShouldStopWithCancel")


if __name__ == "__main__":
    run_tests()
