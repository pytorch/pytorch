import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/math_kernel_test"


class TestMathKernelTest(TestCase):
    cpp_name = "MathKernelTest"

    def test_NativeGroupNorm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NativeGroupNorm")

    def test_NativeLayerNorm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NativeLayerNorm")

    def test_Addr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Addr")

    def test_SiluBackward(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SiluBackward")

    def test_MishBackward(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MishBackward")

    def test_NarrowCopy(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NarrowCopy")

    def test_Bmm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Bmm")


if __name__ == "__main__":
    run_tests()
