import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/common_test"


class TestCommonTest(TestCase):
    cpp_name = "CommonTest"

    def test_TestStoi(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestStoi")

    def test_TestStod(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestStod")


if __name__ == "__main__":
    run_tests()
