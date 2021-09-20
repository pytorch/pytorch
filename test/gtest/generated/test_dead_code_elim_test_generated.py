import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/dead_code_elim_test"


class TestDeadCodeElim(TestCase):
    cpp_name = "DeadCodeElim"

    def test_BasicElim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BasicElim")

    def test_BasicNoElim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BasicNoElim")

    def test_PartiallyUsedNoElim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PartiallyUsedNoElim")


if __name__ == "__main__":
    run_tests()
