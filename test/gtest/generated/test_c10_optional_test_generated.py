import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_optional_test"


class TestOptionalTest_0(TestCase):
    cpp_name = "OptionalTest/0"

    def test_Empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Empty")

    def test_Initialized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Initialized")


class TestOptionalTest_1(TestCase):
    cpp_name = "OptionalTest/1"

    def test_Empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Empty")

    def test_Initialized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Initialized")


class TestOptionalTest_2(TestCase):
    cpp_name = "OptionalTest/2"

    def test_Empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Empty")

    def test_Initialized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Initialized")


class TestOptionalTest_3(TestCase):
    cpp_name = "OptionalTest/3"

    def test_Empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Empty")

    def test_Initialized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Initialized")


class TestOptionalTest(TestCase):
    cpp_name = "OptionalTest"

    def test_Nullopt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Nullopt")


class TestCmpTest_0(TestCase):
    cpp_name = "CmpTest/0"

    def test_Cmp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cmp")


class TestCmpTest_1(TestCase):
    cpp_name = "CmpTest/1"

    def test_Cmp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cmp")


class TestCmpTest_2(TestCase):
    cpp_name = "CmpTest/2"

    def test_Cmp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cmp")


class TestCmpTest_3(TestCase):
    cpp_name = "CmpTest/3"

    def test_Cmp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cmp")


class TestCmpTest_4(TestCase):
    cpp_name = "CmpTest/4"

    def test_Cmp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cmp")


class Testnullopt_SelfCompareTest(TestCase):
    cpp_name = "nullopt/SelfCompareTest"

    def test_SelfCompare_0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SelfCompare/0")


class Testint_SelfCompareTest(TestCase):
    cpp_name = "int/SelfCompareTest"

    def test_SelfCompare_0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SelfCompare/0")


if __name__ == "__main__":
    run_tests()
