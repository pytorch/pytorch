import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/generate_proposals_op_test"


class TestGenerateProposalsTest(TestCase):
    cpp_name = "GenerateProposalsTest"

    def test_TestComputeAllAnchors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestComputeAllAnchors")

    def test_TestComputeSortedAnchors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestComputeSortedAnchors")

    def test_TestComputeAllAnchorsRotated(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestComputeAllAnchorsRotated")

    def test_TestComputeSortedAnchorsRotated(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestComputeSortedAnchorsRotated")

    def test_TestEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEmpty")

    def test_TestRealDownSampled(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestRealDownSampled")

    def test_TestRealDownSampledRotatedAngle0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestRealDownSampledRotatedAngle0")

    def test_TestRealDownSampledRotated(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestRealDownSampledRotated")


if __name__ == "__main__":
    run_tests()
