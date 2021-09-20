import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/TensorImpl_test"


class TestTensorImplTest(TestCase):
    cpp_name = "TensorImplTest"

    def test_Caffe2Constructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Caffe2Constructor")


if __name__ == "__main__":
    run_tests()
