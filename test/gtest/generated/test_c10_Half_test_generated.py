import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_Half_test"


class TestHalfDoubleConversionTest(TestCase):
    cpp_name = "HalfDoubleConversionTest"

    def test_Half2Double(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Half2Double")


if __name__ == "__main__":
    run_tests()
