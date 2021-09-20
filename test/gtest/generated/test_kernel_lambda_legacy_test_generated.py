import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/kernel_lambda_legacy_test"


class TestOperatorRegistrationTest_LegacyLambdaBasedKernel(TestCase):
    cpp_name = "OperatorRegistrationTest_LegacyLambdaBasedKernel"

    def test_givenKernel_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenKernel_whenRegistered_thenCanBeCalled"
        )

    def test_givenKernel_whenRegisteredInConstructor_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernel_whenRegisteredInConstructor_thenCanBeCalled",
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

    def test_givenKernelWithoutOutput_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithoutOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithIntOutput_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithIntOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithTensorInputByReference_withOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithTensorInputByReference_withOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithTensorInputByValue_withOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithTensorInputByValue_withOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithTensorInputByValue_withoutOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithTensorInputByValue_withoutOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithTensorVectorInput_withOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithTensorVectorInput_withOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithLegacyTensorVectorInput_withoutOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithLegacyTensorVectorInput_withoutOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithLegacyTensorVectorInput_withOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithLegacyTensorVectorInput_withOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithLegacyTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithLegacyTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithLegacyTensorListInput_withOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithLegacyTensorListInput_withOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithStringListOutput_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithStringListOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithDictInput_withoutOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithDictInput_withoutOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithDictInput_withOutput_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithDictInput_withOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithDictOutput_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithDictOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithUnorderedMapInput_withoutOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithUnorderedMapInput_withoutOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithUnorderedMapInput_withOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithUnorderedMapInput_withOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithUnorderedMapOutput_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithUnorderedMapOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithMapOfList_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithMapOfList_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithMapOfListOfMap_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithMapOfListOfMap_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithListOfMap_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithListOfMap_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithListOfMapOfIntList_whenRegistered_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithListOfMapOfIntList_whenRegistered_thenCanBeCalled",
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

    def test_givenKernelWithOptionalInputs_withoutOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithOptionalInputs_withoutOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithOptionalInputs_withOutput_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithOptionalInputs_withOutput_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernelWithOptionalInputs_withMultipleOutputs_whenRegistered_thenCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernelWithOptionalInputs_withMultipleOutputs_whenRegistered_thenCanBeCalled",
        )

    def test_givenKernel_whenRegistered_thenCanBeCalledUnboxed(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernel_whenRegistered_thenCanBeCalledUnboxed",
        )

    def test_givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema",
        )

    def test_givenMismatchedKernel_withDifferentNumArguments_whenRegistering_thenFails(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenMismatchedKernel_withDifferentNumArguments_whenRegistering_thenFails",
        )

    def test_givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails",
        )

    def test_givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails",
        )

    def test_givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails",
        )


if __name__ == "__main__":
    run_tests()
