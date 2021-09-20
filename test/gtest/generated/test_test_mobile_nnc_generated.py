import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/test_mobile_nnc"


class TestFunction(TestCase):
    cpp_name = "Function"

    def test_ExecuteSlowMul(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExecuteSlowMul")

    def test_Serialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Serialization")

    def test_ValidInput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ValidInput")

    def test_InvalidInput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InvalidInput")


class TestNNCBackendTest(TestCase):
    cpp_name = "NNCBackendTest"

    def test_AOTCompileThenExecute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AOTCompileThenExecute")

    def test_FakeTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FakeTensor")


class TestMobileNNCRegistryTest(TestCase):
    cpp_name = "MobileNNCRegistryTest"

    def test_FindAndRun(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FindAndRun")

    def test_NoKernel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoKernel")


if __name__ == "__main__":
    run_tests()
