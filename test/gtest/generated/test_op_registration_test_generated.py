import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/op_registration_test"


class TestOperatorRegistrationTest(TestCase):
    cpp_name = "OperatorRegistrationTest"

    def test_whenRegisteringWithSchemaBeforeKernelInOptionsObject_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringWithSchemaBeforeKernelInOptionsObject_thenCanBeCalled",
        )

    def test_whenRegisteringWithSchemaAfterKernelInOptionsObject_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringWithSchemaAfterKernelInOptionsObject_thenCanBeCalled",
        )

    def test_whenRegisteringWithNameBeforeKernelInOptionsObject_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringWithNameBeforeKernelInOptionsObject_thenCanBeCalled",
        )

    def test_whenRegisteringWithNameAfterKernelInOptionsObject_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringWithNameAfterKernelInOptionsObject_thenCanBeCalled",
        )

    def test_whenRegisteringWithoutSchema_thenFails(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenRegisteringWithoutSchema_thenFails"
        )

    def test_whenCallingOpWithWrongDispatchKey_thenFails(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenCallingOpWithWrongDispatchKey_thenFails"
        )

    def test_givenOpWithCatchallKernel_whenCallingOp_thenCallsCatchallKernel(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOpWithCatchallKernel_whenCallingOp_thenCallsCatchallKernel",
        )

    def test_givenOpWithDispatchedKernelOutOfScope_whenRegisteringCatchallKernelAndCallingOp_thenCallsCatchallKernel(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOpWithDispatchedKernelOutOfScope_whenRegisteringCatchallKernelAndCallingOp_thenCallsCatchallKernel",
        )

    def test_givenOpWithCatchallKernelOutOfScope_whenRegisteringDispatchedKernelAndCallingOp_thenCallsCatchallKernel(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOpWithCatchallKernelOutOfScope_whenRegisteringDispatchedKernelAndCallingOp_thenCallsCatchallKernel",
        )

    def test_givenOpWithoutKernels_whenRegisteringWithSchema_thenOnlyRegistersSchema(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOpWithoutKernels_whenRegisteringWithSchema_thenOnlyRegistersSchema",
        )

    def test_givenOpWithoutKernels_whenRegisteringWithoutSchema_thenFails(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOpWithoutKernels_whenRegisteringWithoutSchema_thenFails",
        )

    def test_givenOpWithoutKernels_whenRunningOutOfScope_thenSchemaIsGone(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOpWithoutKernels_whenRunningOutOfScope_thenSchemaIsGone",
        )

    def test_givenOpWithoutKernelsWithoutTensorInputs_whenRegistering_thenRegisters(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOpWithoutKernelsWithoutTensorInputs_whenRegistering_thenRegisters",
        )

    def test_givenMultipleKernelsWithSameDispatchKey_whenRegisteringInSameOpCall_thenFails(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenMultipleKernelsWithSameDispatchKey_whenRegisteringInSameOpCall_thenFails",
        )

    def test_givenMultipleCatchallKernels_whenRegisteringInSameOpCall_thenFails(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenMultipleCatchallKernels_whenRegisteringInSameOpCall_thenFails",
        )

    def test_whenRegisteringCPUTensorType_thenCanOnlyCallUnboxedWithCPUDispatchKey(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringCPUTensorType_thenCanOnlyCallUnboxedWithCPUDispatchKey",
        )

    def test_whenRegisteringMultipleKernelsInSameOpCallAndCalling_thenCallsCorrectKernel(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringMultipleKernelsInSameOpCallAndCalling_thenCallsCorrectKernel",
        )

    def test_whenRegisteringMultipleKernelsByNameAndNoneCanInferSchema_thenFails(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringMultipleKernelsByNameAndNoneCanInferSchema_thenFails",
        )

    def test_whenRegisteringMultipleKernelsBySchemaAndNoneCanInferSchema_thenSucceeds(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringMultipleKernelsBySchemaAndNoneCanInferSchema_thenSucceeds",
        )

    def test_whenRegisteringMultipleKernelsByNameAndOnlyOneCanInferSchema_thenSucceeds(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringMultipleKernelsByNameAndOnlyOneCanInferSchema_thenSucceeds",
        )

    def test_whenRegisteringMultipleKernelsBySchemaAndOnlyOneCanInferSchema_thenSucceeds(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringMultipleKernelsBySchemaAndOnlyOneCanInferSchema_thenSucceeds",
        )

    def test_whenRegisteringMismatchingKernelsInSameOpCall_thenFails(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringMismatchingKernelsInSameOpCall_thenFails",
        )

    def test_whenRegisteringBackendFallbackKernel_thenCanBeCalled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringBackendFallbackKernel_thenCanBeCalled",
        )

    def test_whenRegisteringBackendFallbackKernelForWrongBackend_thenCannotBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringBackendFallbackKernelForWrongBackend_thenCannotBeCalled",
        )

    def test_whenRegisteringBackendFallbackKernelAndRegularKernelForDifferentBackend_thenRegularKernelCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringBackendFallbackKernelAndRegularKernelForDifferentBackend_thenRegularKernelCanBeCalled",
        )

    def test_whenRegisteringBackendFallbackKernelAndRegularKernelForDifferentBackend_thenFallbackKernelCanBeCalled(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringBackendFallbackKernelAndRegularKernelForDifferentBackend_thenFallbackKernelCanBeCalled",
        )

    def test_whenRegisteringBackendFallbackKernelAndRegularKernelForSameBackend_thenCallsRegularKernel(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringBackendFallbackKernelAndRegularKernelForSameBackend_thenCallsRegularKernel",
        )

    def test_whenRegisteringAutogradKernel_thenCanCallAutogradKernel(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringAutogradKernel_thenCanCallAutogradKernel",
        )

    def test_whenRegisteringAutogradKernelWithRegularKernel_thenCanCallAutogradKernel(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringAutogradKernelWithRegularKernel_thenCanCallAutogradKernel",
        )

    def test_whenRegisteringAutogradKernelWithCatchAllKernel_thenCanCallAutogradKernel(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringAutogradKernelWithCatchAllKernel_thenCanCallAutogradKernel",
        )

    def test_whenRegisteringAutogradKernelWithCatchAllKernel_thenCanCallCatchallKernel(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringAutogradKernelWithCatchAllKernel_thenCanCallCatchallKernel",
        )

    def test_AutogradBackendOverridesAutogradKernel(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "AutogradBackendOverridesAutogradKernel"
        )

    def test_AutogradXLAOverridesAutogradKernel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AutogradXLAOverridesAutogradKernel")

    def test_AutogradLazyOverridesAutogradKernel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AutogradLazyOverridesAutogradKernel")

    def test_whenRegisterWithXLAKernelAndCatchAll_AutogradXLAIsNotFilled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisterWithXLAKernelAndCatchAll_AutogradXLAIsNotFilled",
        )

    def test_whenRegisterWithLazyKernelAndCatchAll_AutogradLazyIsNotFilled(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisterWithLazyKernelAndCatchAll_AutogradLazyIsNotFilled",
        )

    def test_givenLambdaKernel_whenRegisteringWithMismatchingCppSignatures_thenFails(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLambdaKernel_whenRegisteringWithMismatchingCppSignatures_thenFails",
        )

    def test_givenLambdaKernel_whenRegisteringCatchAllAndBackendWithMismatchingCppSignatures_thenFails(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLambdaKernel_whenRegisteringCatchAllAndBackendWithMismatchingCppSignatures_thenFails",
        )

    def test_givenLambdaKernel_whenRegisteringBackendAndCatchAllWithMismatchingCppSignatures_thenFails(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLambdaKernel_whenRegisteringBackendAndCatchAllWithMismatchingCppSignatures_thenFails",
        )

    def test_givenLambdaKernel_whenAccessingWithMismatchingCppSignatures_thenFails(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLambdaKernel_whenAccessingWithMismatchingCppSignatures_thenFails",
        )

    def test_givenLambdaKernel_whenAccessingCatchAllWithMismatchingCppSignatures_thenFails(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLambdaKernel_whenAccessingCatchAllWithMismatchingCppSignatures_thenFails",
        )

    def test_givenTorchLibrary_whenRegisteringWithMismatchingCppSignatures_thenFails(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenTorchLibrary_whenRegisteringWithMismatchingCppSignatures_thenFails",
        )

    def test_givenTorchLibrary_whenAccessingWithMismatchingCppSignatures_thenFails(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenTorchLibrary_whenAccessingWithMismatchingCppSignatures_thenFails",
        )

    def test_givenTorchLibrary_whenAccessingCatchAllWithMismatchingCppSignatures_thenFails(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenTorchLibrary_whenAccessingCatchAllWithMismatchingCppSignatures_thenFails",
        )

    def test_testAvailableArgTypes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testAvailableArgTypes")

    def test_callKernelsWithDispatchKeySetConvention_call_redispatchesToLowerPriorityKernels(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "callKernelsWithDispatchKeySetConvention_call_redispatchesToLowerPriorityKernels",
        )

    def test_callKernelsWithDispatchKeySetConvention_callBoxed_redispatchesToLowerPriorityKernels(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "callKernelsWithDispatchKeySetConvention_callBoxed_redispatchesToLowerPriorityKernels",
        )

    def test_callKernelsWithDispatchKeySetConvention_mixedCallingConventions_redispatchesToLowerPriorityKernels(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "callKernelsWithDispatchKeySetConvention_mixedCallingConventions_redispatchesToLowerPriorityKernels",
        )


class TestNewOperatorRegistrationTest(TestCase):
    cpp_name = "NewOperatorRegistrationTest"

    def test_testBasics(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testBasics")

    def test_importTopLevel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "importTopLevel")

    def test_overload(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "overload")

    def test_importNamespace(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "importNamespace")

    def test_schema(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "schema")

    def test_whenRegisteringBackendFallbackKernelAndCatchallKernelForSameBackend_thenCallsFallbackKernel(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringBackendFallbackKernelAndCatchallKernelForSameBackend_thenCallsFallbackKernel",
        )

    def test_whenRegisteringAutogradKernelWithRegularKernel_thenCanCallRegularKernel(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenRegisteringAutogradKernelWithRegularKernel_thenCanCallRegularKernel",
        )

    def test_dispatchWithCompositeImplicitAutogradKernel(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "dispatchWithCompositeImplicitAutogradKernel"
        )

    def test_dispatchWithCompositeImplicitAutogradAndAutogradKernel(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "dispatchWithCompositeImplicitAutogradAndAutogradKernel",
        )

    def test_dispatchWithCompositeImplicitAutogradAndCatchAllKernel(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "dispatchWithCompositeImplicitAutogradAndCatchAllKernel",
        )

    def test_AutogradBackendOverridesCompositeImplicitAutogradKernel(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "AutogradBackendOverridesCompositeImplicitAutogradKernel",
        )

    def test_BackendOverridesCompositeImplicitAutogradKernel(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "BackendOverridesCompositeImplicitAutogradKernel",
        )

    def test_dispatchWithCompositeExplicitAutogradKernel(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "dispatchWithCompositeExplicitAutogradKernel"
        )

    def test_dispatchWithCompositeExplicitAutogradAndCompositeImplicitAutogradKernel(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "dispatchWithCompositeExplicitAutogradAndCompositeImplicitAutogradKernel",
        )

    def test_BackendOverridesCompositeExplicitAutogradKernel(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "BackendOverridesCompositeExplicitAutogradKernel",
        )

    def test_dispatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "dispatch")

    def test_dispatchAutogradPrecedence(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "dispatchAutogradPrecedence")

    def test_throwsWhenRegisterToBackendMapsToAutogradOther(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "throwsWhenRegisterToBackendMapsToAutogradOther"
        )

    def test_dispatchMultipleTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "dispatchMultipleTensors")

    def test_dispatchMultiple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "dispatchMultiple")

    def test_fallback(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fallback")

    def test_BackendSelectRedispatchesToCPU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BackendSelectRedispatchesToCPU")

    def test_TorchLibraryTwiceIsError(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchLibraryTwiceIsError")

    def test_CppFunction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CppFunction")

    def test_testDelayedListener(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testDelayedListener")

    def test_testImplNoDefGetsCaught(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testImplNoDefGetsCaught")


if __name__ == "__main__":
    run_tests()
