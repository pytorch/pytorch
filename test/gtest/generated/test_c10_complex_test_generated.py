import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_complex_test"


class TestTestMemory(TestCase):
    cpp_name = "TestMemory"

    def test_ReinterpretCast(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReinterpretCast")


class TestTestConstructors(TestCase):
    cpp_name = "TestConstructors"

    def test_UnorderedMap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnorderedMap")


class TestTestArithmeticIntScalar(TestCase):
    cpp_name = "TestArithmeticIntScalar"

    def test_All(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "All")


class TestTestIO(TestCase):
    cpp_name = "TestIO"

    def test_All(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "All")


class TestTestStd(TestCase):
    cpp_name = "TestStd"

    def test_BasicFunctions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BasicFunctions")


if __name__ == "__main__":
    run_tests()
