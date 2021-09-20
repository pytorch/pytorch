import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/Dimname_test"


class TestDimnameTest(TestCase):
    cpp_name = "DimnameTest"

    def test_isValidIdentifier(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "isValidIdentifier")

    def test_wildcardName(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "wildcardName")

    def test_createNormalName(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "createNormalName")

    def test_unifyAndMatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "unifyAndMatch")


if __name__ == "__main__":
    run_tests()
