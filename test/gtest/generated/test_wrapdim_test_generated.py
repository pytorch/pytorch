import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/wrapdim_test"


class TestTestWrapdim(TestCase):
    cpp_name = "TestWrapdim"

    def test_TestWrapdim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestWrapdim")


if __name__ == "__main__":
    run_tests()
