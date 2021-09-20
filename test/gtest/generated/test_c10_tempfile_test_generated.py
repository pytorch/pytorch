import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_tempfile_test"


class TestTempFileTest(TestCase):
    cpp_name = "TempFileTest"

    def test_MatchesExpectedPattern(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MatchesExpectedPattern")


class TestTempDirTest(TestCase):
    cpp_name = "TempDirTest"

    def test_tryMakeTempdir(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "tryMakeTempdir")


if __name__ == "__main__":
    run_tests()
