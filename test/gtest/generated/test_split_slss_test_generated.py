import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/split_slss_test"


class TestsplitSparseLengthsSumSparse(TestCase):
    cpp_name = "splitSparseLengthsSumSparse"

    def test_sweep(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "sweep")


if __name__ == "__main__":
    run_tests()
