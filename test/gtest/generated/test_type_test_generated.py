import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/type_test"


class TestTypeCustomPrinter(TestCase):
    cpp_name = "TypeCustomPrinter"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_ContainedTypes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ContainedTypes")

    def test_NamedTuples(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NamedTuples")


class TestTypeEquality(TestCase):
    cpp_name = "TypeEquality"

    def test_ClassBasic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClassBasic")

    def test_ClassInequality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClassInequality")

    def test_InterfaceEquality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InterfaceEquality")

    def test_InterfaceInequality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InterfaceInequality")

    def test_TupleEquality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleEquality")

    def test_NamedTupleEquality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NamedTupleEquality")


if __name__ == "__main__":
    run_tests()
