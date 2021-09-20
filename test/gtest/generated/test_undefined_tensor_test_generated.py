import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/undefined_tensor_test"


class TestTestUndefined(TestCase):
    cpp_name = "TestUndefined"

    def test_UndefinedTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UndefinedTest")


if __name__ == "__main__":
    run_tests()
