import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_bfloat16_test"


class TestBFloat16Conversion(TestCase):
    cpp_name = "BFloat16Conversion"

    def test_FloatToBFloat16AndBack(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FloatToBFloat16AndBack")

    def test_FloatToBFloat16RNEAndBack(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FloatToBFloat16RNEAndBack")

    def test_NaN(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NaN")

    def test_Inf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Inf")

    def test_SmallestDenormal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SmallestDenormal")


class TestBFloat16Math(TestCase):
    cpp_name = "BFloat16Math"

    def test_Addition(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Addition")

    def test_Subtraction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Subtraction")

    def test_NextAfterZero(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NextAfterZero")


class TestBFloat16Test_Instantiation_BFloat16Test(TestCase):
    cpp_name = "BFloat16Test_Instantiation/BFloat16Test"

    def test_BFloat16RNETest_0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BFloat16RNETest/0")

    def test_BFloat16RNETest_1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BFloat16RNETest/1")

    def test_BFloat16RNETest_2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BFloat16RNETest/2")

    def test_BFloat16RNETest_3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BFloat16RNETest/3")

    def test_BFloat16RNETest_4(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BFloat16RNETest/4")


if __name__ == "__main__":
    run_tests()
