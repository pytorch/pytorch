import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/half_test"


class TestTestHalf(TestCase):
    cpp_name = "TestHalf"

    def test_Arithmetic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Arithmetic")

    def test_Comparisions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Comparisions")

    def test_Cast(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cast")

    def test_Construction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Construction")

    def test_Half2String(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Half2String")

    def test_HalfNumericLimits(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HalfNumericLimits")

    def test_CommonMath(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CommonMath")


if __name__ == "__main__":
    run_tests()
