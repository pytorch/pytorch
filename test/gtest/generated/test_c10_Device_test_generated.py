import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_Device_test"


class TestDeviceTest(TestCase):
    cpp_name = "DeviceTest"

    def test_BasicConstruction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BasicConstruction")


if __name__ == "__main__":
    run_tests()
