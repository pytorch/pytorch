import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/utility_ops_test"


class TestUtilityOpTest(TestCase):
    cpp_name = "UtilityOpTest"

    def test_testReshapeWithScalar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testReshapeWithScalar")


if __name__ == "__main__":
    run_tests()
