import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/boolean_unmask_ops_test"


class TestBooleanUnmaskTest(TestCase):
    cpp_name = "BooleanUnmaskTest"

    def test_Test(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Test")


if __name__ == "__main__":
    run_tests()
