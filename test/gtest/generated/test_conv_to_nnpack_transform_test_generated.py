import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/conv_to_nnpack_transform_test"


class TestConvToNNPackTest(TestCase):
    cpp_name = "ConvToNNPackTest"

    def test_TestSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSimple")


if __name__ == "__main__":
    run_tests()
