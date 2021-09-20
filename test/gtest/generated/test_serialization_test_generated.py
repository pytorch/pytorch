import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/serialization_test"


class TestTensorSerialization(TestCase):
    cpp_name = "TensorSerialization"

    def test_TestUnknownDType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestUnknownDType")


if __name__ == "__main__":
    run_tests()
