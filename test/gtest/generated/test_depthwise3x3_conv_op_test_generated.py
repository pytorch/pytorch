import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/depthwise3x3_conv_op_test"


class TestDEPTHWISE3x3(TestCase):
    cpp_name = "DEPTHWISE3x3"

    def test_Conv(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv")


if __name__ == "__main__":
    run_tests()
