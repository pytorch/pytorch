import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/KernelFunction_test"


class TestKernelFunctionTest(TestCase):
    cpp_name = "KernelFunctionTest"

    def test_givenBoxedFunction_withReturn_whenCallingBoxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBoxedFunction_withReturn_whenCallingBoxed_thenWorks",
        )

    def test_givenBoxedFunction_withoutReturn_whenCallingBoxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBoxedFunction_withoutReturn_whenCallingBoxed_thenWorks",
        )

    def test_givenBoxedFunction_withMultiReturn_whenCallingBoxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBoxedFunction_withMultiReturn_whenCallingBoxed_thenWorks",
        )

    def test_givenBoxedFunction_withInPlaceSignature_whenCallingBoxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBoxedFunction_withInPlaceSignature_whenCallingBoxed_thenWorks",
        )

    def test_givenBoxedFunction_withOutOfPlaceSignature_whenCallingBoxed_thenWorks(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBoxedFunction_withOutOfPlaceSignature_whenCallingBoxed_thenWorks",
        )

    def test_givenBoxedFunction_withOutOfPlaceMultiSignature_whenCallingBoxed_thenWorks(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBoxedFunction_withOutOfPlaceMultiSignature_whenCallingBoxed_thenWorks",
        )

    def test_givenBoxedFunction_withReturn_whenCallingUnboxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBoxedFunction_withReturn_whenCallingUnboxed_thenWorks",
        )

    def test_givenBoxedFunction_withoutReturn_whenCallingUnboxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBoxedFunction_withoutReturn_whenCallingUnboxed_thenWorks",
        )

    def test_givenBoxedFunction_withMultiReturn_whenCallingUnboxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBoxedFunction_withMultiReturn_whenCallingUnboxed_thenWorks",
        )

    def test_givenBoxedFunction_withInPlaceSignature_whenCallingUnboxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBoxedFunction_withInPlaceSignature_whenCallingUnboxed_thenWorks",
        )

    def test_givenBoxedFunction_withOutOfPlaceSignature_whenCallingUnboxed_thenWorks(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBoxedFunction_withOutOfPlaceSignature_whenCallingUnboxed_thenWorks",
        )

    def test_givenBoxedFunction_withOutOfPlaceMultiSignature_whenCallingUnboxed_thenWorks(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBoxedFunction_withOutOfPlaceMultiSignature_whenCallingUnboxed_thenWorks",
        )

    def test_givenUnboxedFunctor_withReturn_whenCallingBoxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedFunctor_withReturn_whenCallingBoxed_thenWorks",
        )

    def test_givenUnboxedFunctor_withoutReturn_whenCallingBoxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedFunctor_withoutReturn_whenCallingBoxed_thenWorks",
        )

    def test_givenUnboxedFunctor_withReturn_whenCallingUnboxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedFunctor_withReturn_whenCallingUnboxed_thenWorks",
        )

    def test_givenUnboxedFunctor_withoutReturn_whenCallingUnboxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedFunctor_withoutReturn_whenCallingUnboxed_thenWorks",
        )

    def test_givenUnboxedFunction_withReturn_whenCallingBoxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedFunction_withReturn_whenCallingBoxed_thenWorks",
        )

    def test_givenUnboxedFunction_withoutReturn_whenCallingBoxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedFunction_withoutReturn_whenCallingBoxed_thenWorks",
        )

    def test_givenUnboxedFunction_withReturn_whenCallingUnboxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedFunction_withReturn_whenCallingUnboxed_thenWorks",
        )

    def test_givenUnboxedFunction_withoutReturn_whenCallingUnboxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedFunction_withoutReturn_whenCallingUnboxed_thenWorks",
        )

    def test_givenUnboxedRuntimeFunction_withReturn_whenCallingBoxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedRuntimeFunction_withReturn_whenCallingBoxed_thenWorks",
        )

    def test_givenUnboxedRuntimeFunction_withoutReturn_whenCallingBoxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedRuntimeFunction_withoutReturn_whenCallingBoxed_thenWorks",
        )

    def test_givenUnboxedRuntimeFunction_withReturn_whenCallingUnboxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedRuntimeFunction_withReturn_whenCallingUnboxed_thenWorks",
        )

    def test_givenUnboxedRuntimeFunction_withoutReturn_whenCallingUnboxed_thenWorks(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedRuntimeFunction_withoutReturn_whenCallingUnboxed_thenWorks",
        )

    def test_givenUnboxedLambda_withReturn_whenCallingBoxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedLambda_withReturn_whenCallingBoxed_thenWorks",
        )

    def test_givenUnboxedLambda_withoutReturn_whenCallingBoxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedLambda_withoutReturn_whenCallingBoxed_thenWorks",
        )

    def test_givenUnboxedLambda_withReturn_whenCallingUnboxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedLambda_withReturn_whenCallingUnboxed_thenWorks",
        )

    def test_givenUnboxedLambda_withoutReturn_whenCallingUnboxed_thenWorks(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenUnboxedLambda_withoutReturn_whenCallingUnboxed_thenWorks",
        )


if __name__ == "__main__":
    run_tests()
