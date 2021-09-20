import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/math_test"


class TestMathTest(TestCase):
    cpp_name = "MathTest"

    def test_GemmNoTransNoTrans(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GemmNoTransNoTrans")

    def test_GemmNoTransTrans(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GemmNoTransTrans")

    def test_GemvNoTrans(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GemvNoTrans")

    def test_GemvTrans(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GemvTrans")

    def test_FloatToHalfConversion(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FloatToHalfConversion")


class TestBroadcastTest(TestCase):
    cpp_name = "BroadcastTest"

    def test_BroadcastFloatTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BroadcastFloatTest")


class TestRandFixedSumTest(TestCase):
    cpp_name = "RandFixedSumTest"

    def test_UpperBound(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UpperBound")


class TestGemmBatchedTrans_GemmBatchedTest(TestCase):
    cpp_name = "GemmBatchedTrans/GemmBatchedTest"

    def test_GemmBatchedFloatTest_0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GemmBatchedFloatTest/0")

    def test_GemmBatchedFloatTest_1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GemmBatchedFloatTest/1")

    def test_GemmBatchedFloatTest_2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GemmBatchedFloatTest/2")

    def test_GemmBatchedFloatTest_3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GemmBatchedFloatTest/3")

    def test_GemmStridedBatchedFloatTest_0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GemmStridedBatchedFloatTest/0")

    def test_GemmStridedBatchedFloatTest_1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GemmStridedBatchedFloatTest/1")

    def test_GemmStridedBatchedFloatTest_2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GemmStridedBatchedFloatTest/2")

    def test_GemmStridedBatchedFloatTest_3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GemmStridedBatchedFloatTest/3")


if __name__ == "__main__":
    run_tests()
