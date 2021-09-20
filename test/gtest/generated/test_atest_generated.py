import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/atest"


class Testatest(TestCase):
    cpp_name = "atest"

    def test_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "operators")

    def test_logical_and_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "logical_and_operators")

    def test_logical_or_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "logical_or_operators")

    def test_logical_xor_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "logical_xor_operators")

    def test_lt_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "lt_operators")

    def test_le_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "le_operators")

    def test_gt_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "gt_operators")

    def test_ge_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ge_operators")

    def test_eq_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "eq_operators")

    def test_ne_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ne_operators")

    def test_add_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "add_operators")

    def test_max_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "max_operators")

    def test_min_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "min_operators")

    def test_sigmoid_backward_operator(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "sigmoid_backward_operator")

    def test_fmod_tensor_operators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fmod_tensor_operators")

    def test_atest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "atest")


if __name__ == "__main__":
    run_tests()
