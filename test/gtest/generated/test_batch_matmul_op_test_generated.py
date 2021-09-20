import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/batch_matmul_op_test"


class TestBatchMatMulOpTest(TestCase):
    cpp_name = "BatchMatMulOpTest"

    def test_BatchMatMulOpNormalTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchMatMulOpNormalTest")

    def test_BatchMatMulOpBroadcastTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchMatMulOpBroadcastTest")


if __name__ == "__main__":
    run_tests()
