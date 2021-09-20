import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/transform_test"


class TestTransformTest(TestCase):
    cpp_name = "TransformTest"

    def test_TestPatternMatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestPatternMatch")

    def test_TestReplacePattern(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestReplacePattern")

    def test_TestTransformApply(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestTransformApply")

    def test_TestPatternMatchTypeSortedOrder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestPatternMatchTypeSortedOrder")

    def test_TestPatternMatchTypeGeneral(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestPatternMatchTypeGeneral")

    def test_TestApplyTransformIfFasterIsFaster(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestApplyTransformIfFasterIsFaster")

    def test_TestApplyTransformIfFasterButSlower(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestApplyTransformIfFasterButSlower")


if __name__ == "__main__":
    run_tests()
