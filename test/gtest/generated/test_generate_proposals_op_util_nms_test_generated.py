import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/generate_proposals_op_util_nms_test"


class TestUtilsNMSTest(TestCase):
    cpp_name = "UtilsNMSTest"

    def test_TestNMS(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNMS")

    def test_TestNMS1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNMS1")

    def test_TestSoftNMS(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSoftNMS")

    def test_TestNMSRotatedAngle0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNMSRotatedAngle0")

    def test_TestSoftNMSRotatedAngle0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSoftNMSRotatedAngle0")

    def test_RotatedBBoxOverlaps(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RotatedBBoxOverlaps")


if __name__ == "__main__":
    run_tests()
