import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/int8_roi_align_op_test"


class TestInt8RoIAlign(TestCase):
    cpp_name = "Int8RoIAlign"

    def test_RoIAlign(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RoIAlign")


if __name__ == "__main__":
    run_tests()
