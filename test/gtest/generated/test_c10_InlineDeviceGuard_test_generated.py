import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_InlineDeviceGuard_test"


class TestInlineDeviceGuard(TestCase):
    cpp_name = "InlineDeviceGuard"

    def test_Constructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Constructor")

    def test_ConstructorError(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructorError")

    def test_SetDevice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetDevice")

    def test_ResetDevice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ResetDevice")

    def test_SetIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetIndex")


class TestInlineOptionalDeviceGuard(TestCase):
    cpp_name = "InlineOptionalDeviceGuard"

    def test_Constructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Constructor")

    def test_NullaryConstructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NullaryConstructor")

    def test_SetDevice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetDevice")

    def test_SetIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetIndex")


if __name__ == "__main__":
    run_tests()
