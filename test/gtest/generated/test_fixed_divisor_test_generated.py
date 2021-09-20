import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/fixed_divisor_test"


class TestFixedDivisorTest(TestCase):
    cpp_name = "FixedDivisorTest"

    def test_FixedDivisorInt32Test(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FixedDivisorInt32Test")


if __name__ == "__main__":
    run_tests()
