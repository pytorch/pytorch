import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/observer_test"


class TestObserverTest(TestCase):
    cpp_name = "ObserverTest"

    def test_TestNotify(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNotify")

    def test_TestUniqueMap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestUniqueMap")

    def test_TestNotifyAfterDetach(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNotifyAfterDetach")

    def test_TestDAGNetBase(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDAGNetBase")


if __name__ == "__main__":
    run_tests()
