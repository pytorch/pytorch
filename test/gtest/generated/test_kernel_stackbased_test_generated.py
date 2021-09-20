import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/kernel_stackbased_test"


class TestOperatorRegistrationTest_StackBasedKernel(TestCase):
    cpp_name = "OperatorRegistrationTest_StackBasedKernel"

    def test_givenKernel_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenKernel_whenRegistered_thenCanBeCalled"
        )

    def test_givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel",
        )

    def test_givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel",
        )

    def test_givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore",
        )

    def test_givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled",
        )

    def test_givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernel_whenRegisteredWithoutSpecifyingSchema_thenFailsBecauseItCannotInferFromStackBasedKernel(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernel_whenRegisteredWithoutSpecifyingSchema_thenFailsBecauseItCannotInferFromStackBasedKernel",
        )

    def test_givenKernel_whenRegistered_thenCanAlsoBeCalledUnboxed(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernel_whenRegistered_thenCanAlsoBeCalledUnboxed",
        )

    def test_callKernelsWithDispatchKeySetConvention_redispatchesToLowerPriorityKernels(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "callKernelsWithDispatchKeySetConvention_redispatchesToLowerPriorityKernels",
        )


if __name__ == "__main__":
    run_tests()
