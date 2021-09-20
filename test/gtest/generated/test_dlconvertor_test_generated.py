import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/dlconvertor_test"


class TestTestDlconvertor(TestCase):
    cpp_name = "TestDlconvertor"

    def test_TestDlconvertor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDlconvertor")

    def test_TestDlconvertorNoStrides(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDlconvertorNoStrides")


if __name__ == "__main__":
    run_tests()
