import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_flags_test"


class TestFlagsTest(TestCase):
    cpp_name = "FlagsTest"

    def test_TestGflagsCorrectness(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestGflagsCorrectness")


if __name__ == "__main__":
    run_tests()
