import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_irange_test"


class Testirange_test(TestCase):
    cpp_name = "irange_test"

    def test_range_test(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "range_test")

    def test_end_test(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "end_test")

    def test_neg_range_test(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "neg_range_test")


class Testirange(TestCase):
    cpp_name = "irange"

    def test_empty_reverse_range_two_inputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "empty_reverse_range_two_inputs")

    def test_empty_reverse_range_one_input(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "empty_reverse_range_one_input")


if __name__ == "__main__":
    run_tests()
