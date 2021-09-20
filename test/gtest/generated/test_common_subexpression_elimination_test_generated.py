import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/common_subexpression_elimination_test"


class TestCommonSubexpressionEliminationTest(TestCase):
    cpp_name = "CommonSubexpressionEliminationTest"

    def test_TestSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSimple")

    def test_TestFromExternal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestFromExternal")


if __name__ == "__main__":
    run_tests()
