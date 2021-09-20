import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_accumulate_test"


class Testaccumulate_test(TestCase):
    cpp_name = "accumulate_test"

    def test_vector_test(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "vector_test")

    def test_list_test(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "list_test")

    def test_base_cases(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "base_cases")

    def test_errors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "errors")


if __name__ == "__main__":
    run_tests()
