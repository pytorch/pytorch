import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_complex_math_test"


class TestTestExponential(TestCase):
    cpp_name = "TestExponential"

    def test_IPi(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IPi")

    def test_EulerFormula(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EulerFormula")


class TestTestLog(TestCase):
    cpp_name = "TestLog"

    def test_Definition(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Definition")


class TestTestLog10(TestCase):
    cpp_name = "TestLog10"

    def test_Rev(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Rev")


class TestTestLog2(TestCase):
    cpp_name = "TestLog2"

    def test_Rev(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Rev")


class TestTestPowSqrt(TestCase):
    cpp_name = "TestPowSqrt"

    def test_Equal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Equal")


class TestTestPow(TestCase):
    cpp_name = "TestPow"

    def test_Square(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Square")


class TestTestSinCosSinhCosh(TestCase):
    cpp_name = "TestSinCosSinhCosh"

    def test_Identity(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Identity")


class TestTestTan(TestCase):
    cpp_name = "TestTan"

    def test_Identity(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Identity")


class TestTestTanh(TestCase):
    cpp_name = "TestTanh"

    def test_Identity(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Identity")


class TestTestRevTrigonometric(TestCase):
    cpp_name = "TestRevTrigonometric"

    def test_Rev(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Rev")


class TestTestRevHyperbolic(TestCase):
    cpp_name = "TestRevHyperbolic"

    def test_Rev(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Rev")


if __name__ == "__main__":
    run_tests()
