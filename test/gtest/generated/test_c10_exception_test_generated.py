import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_exception_test"


class TestExceptionTest(TestCase):
    cpp_name = "ExceptionTest"

    def test_TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TORCH_INTERNAL_ASSERT_DEBUG_ONLY")

    def test_CUDA_KERNEL_ASSERT(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CUDA_KERNEL_ASSERT")

    def test_ErrorFormatting(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ErrorFormatting")

    def test_DontCallArgumentFunctionsTwiceOnFailure(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "DontCallArgumentFunctionsTwiceOnFailure"
        )


class TestWarningTest(TestCase):
    cpp_name = "WarningTest"

    def test_JustPrintWarning(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "JustPrintWarning")


if __name__ == "__main__":
    run_tests()
