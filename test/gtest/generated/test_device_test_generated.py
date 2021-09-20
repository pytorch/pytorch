import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/device_test"


class TestDeviceTest(TestCase):
    cpp_name = "DeviceTest"

    def test_InsertCopies(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InsertCopies")


if __name__ == "__main__":
    run_tests()
