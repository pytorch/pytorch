import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/string_ops_test"


class TestStringJoinOpTest(TestCase):
    cpp_name = "StringJoinOpTest"

    def test_testString1DJoin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testString1DJoin")

    def test_testString2DJoin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testString2DJoin")

    def test_testFloat1DJoin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testFloat1DJoin")

    def test_testFloat2DJoin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testFloat2DJoin")

    def test_testLong2DJoin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testLong2DJoin")


if __name__ == "__main__":
    run_tests()
