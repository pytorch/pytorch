import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/weakref_test"


class TestTestWeakPointer(TestCase):
    cpp_name = "TestWeakPointer"

    def test_WeakPointerGetsInvalidated(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "WeakPointerGetsInvalidated")

    def test_WeakPointerLock(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "WeakPointerLock")

    def test_WeakUpdatesRefcountsTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "WeakUpdatesRefcountsTest")


if __name__ == "__main__":
    run_tests()
