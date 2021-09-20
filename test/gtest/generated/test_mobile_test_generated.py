import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/mobile_test"


class TestMobileTest(TestCase):
    cpp_name = "MobileTest"

    def test_Convolution(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Convolution")


if __name__ == "__main__":
    run_tests()
