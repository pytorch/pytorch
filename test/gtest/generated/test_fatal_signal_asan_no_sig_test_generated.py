import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/fatal_signal_asan_no_sig_test"


class TestfatalSignalTest(TestCase):
    cpp_name = "fatalSignalTest"

    def test_SIGABRT8(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SIGABRT8")

    def test_SIGINT8(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SIGINT8")

    def test_SIGILL8(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SIGILL8")

    def test_SIGFPE8(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SIGFPE8")

    def test_SIGBUS8(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SIGBUS8")

    def test_SIGSEGV8(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SIGSEGV8")

    def test_SIGABRT8_NOPRINT(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SIGABRT8_NOPRINT")


if __name__ == "__main__":
    run_tests()
