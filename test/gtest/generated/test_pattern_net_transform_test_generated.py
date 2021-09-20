import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/pattern_net_transform_test"


class TestPatternNetTransformTest(TestCase):
    cpp_name = "PatternNetTransformTest"

    def test_TestGenerateTransform(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestGenerateTransform")

    def test_TestRepeatedTransform(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestRepeatedTransform")

    def test_TestHardTransform(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestHardTransform")

    def test_TestGeneralStringMatching(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestGeneralStringMatching")

    def test_TestDeviceOptionMatching(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDeviceOptionMatching")

    def test_TestEngineMatching(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEngineMatching")

    def test_TestSingularArgumentMatching(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSingularArgumentMatching")

    def test_TestNonStrictTopographicTransform(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNonStrictTopographicTransform")

    def test_TestMultiInputOutputTransform(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestMultiInputOutputTransform")


if __name__ == "__main__":
    run_tests()
