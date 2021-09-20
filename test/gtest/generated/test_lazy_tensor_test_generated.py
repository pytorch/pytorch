import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/lazy_tensor_test"


class TestXlaTensorTest(TestCase):
    cpp_name = "XlaTensorTest"

    def test_TestNoStorage(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNoStorage")


class TestLazyTensorTest(TestCase):
    cpp_name = "LazyTensorTest"

    def test_TestNoStorage(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNoStorage")


if __name__ == "__main__":
    run_tests()
