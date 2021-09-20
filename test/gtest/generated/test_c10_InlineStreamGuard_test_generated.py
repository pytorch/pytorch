import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_InlineStreamGuard_test"


class TestInlineStreamGuard(TestCase):
    cpp_name = "InlineStreamGuard"

    def test_Constructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Constructor")

    def test_ResetStreamSameSameDevice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ResetStreamSameSameDevice")

    def test_ResetStreamDifferentSameDevice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ResetStreamDifferentSameDevice")

    def test_ResetStreamDifferentDevice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ResetStreamDifferentDevice")


class TestInlineOptionalStreamGuard(TestCase):
    cpp_name = "InlineOptionalStreamGuard"

    def test_Constructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Constructor")

    def test_ResetStreamSameDevice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ResetStreamSameDevice")

    def test_ResetStreamDifferentDevice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ResetStreamDifferentDevice")


class TestInlineMultiStreamGuard(TestCase):
    cpp_name = "InlineMultiStreamGuard"

    def test_Constructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Constructor")


if __name__ == "__main__":
    run_tests()
