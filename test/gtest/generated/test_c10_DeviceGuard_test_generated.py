import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_DeviceGuard_test"


class TestDeviceGuard(TestCase):
    cpp_name = "DeviceGuard"

    def test_ResetDeviceDifferentDeviceType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ResetDeviceDifferentDeviceType")


class TestOptionalDeviceGuard(TestCase):
    cpp_name = "OptionalDeviceGuard"

    def test_ResetDeviceDifferentDeviceType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ResetDeviceDifferentDeviceType")


if __name__ == "__main__":
    run_tests()
