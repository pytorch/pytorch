import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/variant_test"


class TestVariantTest(TestCase):
    cpp_name = "VariantTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


if __name__ == "__main__":
    run_tests()
