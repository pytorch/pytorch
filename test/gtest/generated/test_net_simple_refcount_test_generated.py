import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/net_simple_refcount_test"


class TestNetSimpleRefCountTest(TestCase):
    cpp_name = "NetSimpleRefCountTest"

    def test_TestCorrectness(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestCorrectness")


if __name__ == "__main__":
    run_tests()
