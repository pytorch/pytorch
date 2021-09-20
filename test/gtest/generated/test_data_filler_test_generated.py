import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/data_filler_test"


class TestDataFiller(TestCase):
    cpp_name = "DataFiller"

    def test_FillNetInputTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FillNetInputTest")


if __name__ == "__main__":
    run_tests()
