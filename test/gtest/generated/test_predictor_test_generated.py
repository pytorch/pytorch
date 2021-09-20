import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/predictor_test"


class TestPredictorTest(TestCase):
    cpp_name = "PredictorTest"

    def test_SimpleBatchSized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimpleBatchSized")

    def test_SimpleBatchSizedMapInput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimpleBatchSizedMapInput")


if __name__ == "__main__":
    run_tests()
