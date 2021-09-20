import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/generate_proposals_op_util_boxes_test"


class TestUtilsBoxesTest(TestCase):
    cpp_name = "UtilsBoxesTest"

    def test_TestBboxTransformRandom(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBboxTransformRandom")

    def test_TestBboxTransformRotated(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBboxTransformRotated")

    def test_TestBboxTransformRotatedNormalized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBboxTransformRotatedNormalized")

    def test_ClipRotatedBoxes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClipRotatedBoxes")


if __name__ == "__main__":
    run_tests()
