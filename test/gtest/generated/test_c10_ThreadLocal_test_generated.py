import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_ThreadLocal_test"


class TestThreadLocal(TestCase):
    cpp_name = "ThreadLocal"

    def test_TestNoOpScopeWithOneVar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNoOpScopeWithOneVar")


class TestThreadLocalTest(TestCase):
    cpp_name = "ThreadLocalTest"

    def test_TestNoOpScopeWithTwoVars(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNoOpScopeWithTwoVars")

    def test_TestScopeWithOneVar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestScopeWithOneVar")

    def test_TestScopeWithTwoVars(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestScopeWithTwoVars")

    def test_TestInnerScopeWithTwoVars(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestInnerScopeWithTwoVars")

    def test_TestClassScope(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestClassScope")

    def test_TestTwoGlobalScopeVars(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestTwoGlobalScopeVars")

    def test_TestGlobalWithLocalScopeVars(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestGlobalWithLocalScopeVars")

    def test_TestThreadWithLocalScopeVar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestThreadWithLocalScopeVar")

    def test_TestThreadWithGlobalScopeVar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestThreadWithGlobalScopeVar")

    def test_TestObjectsAreReleased(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestObjectsAreReleased")

    def test_TestObjectsAreReleasedByNonstaticThreadLocal(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestObjectsAreReleasedByNonstaticThreadLocal"
        )


if __name__ == "__main__":
    run_tests()
