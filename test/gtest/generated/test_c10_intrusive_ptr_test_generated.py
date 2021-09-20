import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_intrusive_ptr_test"


class TestMakeIntrusiveTest(TestCase):
    cpp_name = "MakeIntrusiveTest"

    def test_ClassWith0Parameters(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClassWith0Parameters")

    def test_ClassWith1Parameter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClassWith1Parameter")

    def test_ClassWith2Parameters(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClassWith2Parameters")

    def test_TypeIsAutoDeductible(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TypeIsAutoDeductible")

    def test_CanAssignToBaseClassPtr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanAssignToBaseClassPtr")


class TestIntrusivePtrTargetTest(TestCase):
    cpp_name = "IntrusivePtrTargetTest"

    def test_whenAllocatedOnStack_thenDoesntCrash(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenAllocatedOnStack_thenDoesntCrash")


class TestIntrusivePtrTest(TestCase):
    cpp_name = "IntrusivePtrTest"

    def test_givenValidPtr_whenCallingGet_thenReturnsObject(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenValidPtr_whenCallingGet_thenReturnsObject"
        )

    def test_givenValidPtr_whenCallingConstGet_thenReturnsObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenCallingConstGet_thenReturnsObject",
        )

    def test_givenInvalidPtr_whenCallingGet_thenReturnsNullptr(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenCallingGet_thenReturnsNullptr",
        )

    def test_givenValidPtr_whenDereferencing_thenReturnsObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenDereferencing_thenReturnsObject",
        )

    def test_givenValidPtr_whenConstDereferencing_thenReturnsObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenConstDereferencing_thenReturnsObject",
        )

    def test_givenValidPtr_whenArrowDereferencing_thenReturnsObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenArrowDereferencing_thenReturnsObject",
        )

    def test_givenValidPtr_whenConstArrowDereferencing_thenReturnsObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenConstArrowDereferencing_thenReturnsObject",
        )

    def test_givenValidPtr_whenMoveAssigning_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigning_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenMoveAssigning_thenOldInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigning_thenOldInstanceInvalid",
        )

    def test_givenValidPtr_whenMoveAssigningToSelf_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigningToSelf_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenMoveAssigningToSelf_thenStaysValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigningToSelf_thenStaysValid",
        )

    def test_givenInvalidPtr_whenMoveAssigningToSelf_thenStaysInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenMoveAssigningToSelf_thenStaysInvalid",
        )

    def test_givenInvalidPtr_whenMoveAssigning_thenNewInstanceIsValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenMoveAssigning_thenNewInstanceIsValid",
        )

    def test_givenInvalidPtr_whenMoveAssigning_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenMoveAssigning_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenMoveAssigningFromInvalidPtr_thenNewInstanceIsInvalid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigningFromInvalidPtr_thenNewInstanceIsInvalid",
        )

    def test_givenValidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenMoveAssigningToBaseClass_thenOldInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigningToBaseClass_thenOldInstanceInvalid",
        )

    def test_givenInvalidPtr_whenMoveAssigningToBaseClass_thenNewInstanceIsValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenMoveAssigningToBaseClass_thenNewInstanceIsValid",
        )

    def test_givenInvalidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject",
        )

    def test_givenInvalidPtr_whenMoveAssigningInvalidPtrToBaseClass_thenNewInstanceIsValid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenMoveAssigningInvalidPtrToBaseClass_thenNewInstanceIsValid",
        )

    def test_givenNullPtr_whenMoveAssigningToDifferentNullptr_thenHasNewNullptr(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNullPtr_whenMoveAssigningToDifferentNullptr_thenHasNewNullptr",
        )

    def test_givenValidPtr_whenCopyAssigning_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenCopyAssigning_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenCopyAssigning_thenOldInstanceValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenCopyAssigning_thenOldInstanceValid",
        )

    def test_givenValidPtr_whenCopyAssigningToSelf_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenCopyAssigningToSelf_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenCopyAssigningToSelf_thenStaysValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenCopyAssigningToSelf_thenStaysValid",
        )

    def test_givenInvalidPtr_whenCopyAssigningToSelf_thenStaysInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenCopyAssigningToSelf_thenStaysInvalid",
        )

    def test_givenInvalidPtr_whenCopyAssigning_thenNewInstanceIsValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenCopyAssigning_thenNewInstanceIsValid",
        )

    def test_givenValidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenCopyAssigningToBaseClass_thenOldInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenCopyAssigningToBaseClass_thenOldInstanceInvalid",
        )

    def test_givenInvalidPtr_whenCopyAssigningToBaseClass_thenNewInstanceIsValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenCopyAssigningToBaseClass_thenNewInstanceIsValid",
        )

    def test_givenInvalidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject",
        )

    def test_givenPtr_whenCopyAssigningInvalidPtrToBaseClass_thenNewInstanceIsInvalid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssigningInvalidPtrToBaseClass_thenNewInstanceIsInvalid",
        )

    def test_givenNullPtr_whenCopyAssigningToDifferentNullptr_thenHasNewNullptr(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNullPtr_whenCopyAssigningToDifferentNullptr_thenHasNewNullptr",
        )

    def test_givenPtr_whenMoveConstructing_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructing_thenPointsToSameObject",
        )

    def test_givenPtr_whenMoveConstructing_thenOldInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructing_thenOldInstanceInvalid",
        )

    def test_givenPtr_whenMoveConstructing_thenNewInstanceValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructing_thenNewInstanceValid",
        )

    def test_givenPtr_whenMoveConstructingFromInvalidPtr_thenNewInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructingFromInvalidPtr_thenNewInstanceInvalid",
        )

    def test_givenPtr_whenMoveConstructingToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructingToBaseClass_thenPointsToSameObject",
        )

    def test_givenPtr_whenMoveConstructingToBaseClass_thenOldInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructingToBaseClass_thenOldInstanceInvalid",
        )

    def test_givenPtr_whenMoveConstructingToBaseClass_thenNewInstanceValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructingToBaseClass_thenNewInstanceValid",
        )

    def test_givenPtr_whenMoveConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid",
        )

    def test_givenNullPtr_whenMoveConstructingToDifferentNullptr_thenHasNewNullptr(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNullPtr_whenMoveConstructingToDifferentNullptr_thenHasNewNullptr",
        )

    def test_givenPtr_whenCopyConstructing_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructing_thenPointsToSameObject",
        )

    def test_givenPtr_whenCopyConstructing_thenOldInstanceValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructing_thenOldInstanceValid",
        )

    def test_givenPtr_whenCopyConstructing_thenNewInstanceValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructing_thenNewInstanceValid",
        )

    def test_givenPtr_whenCopyConstructingFromInvalidPtr_thenNewInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructingFromInvalidPtr_thenNewInstanceInvalid",
        )

    def test_givenPtr_whenCopyConstructingToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructingToBaseClass_thenPointsToSameObject",
        )

    def test_givenPtr_whenCopyConstructingToBaseClass_thenOldInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructingToBaseClass_thenOldInstanceInvalid",
        )

    def test_givenPtr_whenCopyConstructingToBaseClass_thenNewInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructingToBaseClass_thenNewInstanceInvalid",
        )

    def test_givenPtr_whenCopyConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid",
        )

    def test_givenNullPtr_whenCopyConstructingToDifferentNullptr_thenHasNewNullptr(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNullPtr_whenCopyConstructingToDifferentNullptr_thenHasNewNullptr",
        )

    def test_SwapFunction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapFunction")

    def test_SwapMethod(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapMethod")

    def test_SwapFunctionFromInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapFunctionFromInvalid")

    def test_SwapMethodFromInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapMethodFromInvalid")

    def test_SwapFunctionWithInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapFunctionWithInvalid")

    def test_SwapMethodWithInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapMethodWithInvalid")

    def test_SwapFunctionInvalidWithInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapFunctionInvalidWithInvalid")

    def test_SwapMethodInvalidWithInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapMethodInvalidWithInvalid")

    def test_CanBePutInContainer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanBePutInContainer")

    def test_CanBePutInSet(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanBePutInSet")

    def test_CanBePutInUnorderedSet(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanBePutInUnorderedSet")

    def test_CanBePutInMap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanBePutInMap")

    def test_CanBePutInUnorderedMap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanBePutInUnorderedMap")

    def test_Equality_AfterCopyConstructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Equality_AfterCopyConstructor")

    def test_Equality_AfterCopyAssignment(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Equality_AfterCopyAssignment")

    def test_Equality_Nullptr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Equality_Nullptr")

    def test_Inequality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Inequality")

    def test_Inequality_NullptrLeft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Inequality_NullptrLeft")

    def test_Inequality_NullptrRight(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Inequality_NullptrRight")

    def test_HashIsDifferent(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashIsDifferent")

    def test_HashIsDifferent_ValidAndInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashIsDifferent_ValidAndInvalid")

    def test_HashIsSame_AfterCopyConstructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashIsSame_AfterCopyConstructor")

    def test_HashIsSame_AfterCopyAssignment(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashIsSame_AfterCopyAssignment")

    def test_HashIsSame_BothNullptr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashIsSame_BothNullptr")

    def test_OneIsLess(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OneIsLess")

    def test_NullptrIsLess1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NullptrIsLess1")

    def test_NullptrIsLess2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NullptrIsLess2")

    def test_NullptrIsNotLessThanNullptr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NullptrIsNotLessThanNullptr")

    def test_givenPtr_whenCallingReset_thenIsInvalid(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenPtr_whenCallingReset_thenIsInvalid"
        )

    def test_givenPtr_whenCallingReset_thenHoldsNullptr(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenPtr_whenCallingReset_thenHoldsNullptr"
        )

    def test_givenPtr_whenDestructed_thenDestructsObject(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenPtr_whenDestructed_thenDestructsObject"
        )

    def test_givenPtr_whenMoveConstructed_thenDestructsObjectAfterSecondDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructed_thenDestructsObjectAfterSecondDestructed",
        )

    def test_givenPtr_whenMoveConstructedToBaseClass_thenDestructsObjectAfterSecondDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructedToBaseClass_thenDestructsObjectAfterSecondDestructed",
        )

    def test_givenPtr_whenMoveAssigned_thenDestructsOldObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveAssigned_thenDestructsOldObject",
        )

    def test_givenPtr_whenMoveAssignedToBaseClass_thenDestructsOldObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveAssignedToBaseClass_thenDestructsOldObject",
        )

    def test_givenPtrWithCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed",
        )

    def test_givenPtrWithBaseClassCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithBaseClassCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed",
        )

    def test_givenPtrWithCopy_whenMoveAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithCopy_whenMoveAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed",
        )

    def test_givenPtr_whenMoveAssigned_thenDestructsObjectAfterSecondDestructed(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveAssigned_thenDestructsObjectAfterSecondDestructed",
        )

    def test_givenPtr_whenMoveAssignedToBaseClass_thenDestructsObjectAfterSecondDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveAssignedToBaseClass_thenDestructsObjectAfterSecondDestructed",
        )

    def test_givenPtr_whenCopyConstructedAndDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructedAndDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyConstructedToBaseClassAndDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructedToBaseClassAndDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyConstructedAndOriginalDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructedAndOriginalDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyConstructedToBaseClassAndOriginalDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructedToBaseClassAndOriginalDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyAssignedAndDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssignedAndDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyAssignedToBaseClassAndDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssignedToBaseClassAndDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyAssignedAndOriginalDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssignedAndOriginalDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyAssignedToBaseClassAndOriginalDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssignedToBaseClassAndOriginalDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyAssigned_thenDestructsOldObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssigned_thenDestructsOldObject",
        )

    def test_givenPtr_whenCopyAssignedToBaseClass_thenDestructsOldObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssignedToBaseClass_thenDestructsOldObject",
        )

    def test_givenPtrWithCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed",
        )

    def test_givenPtrWithBaseClassCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithBaseClassCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed",
        )

    def test_givenPtrWithCopy_whenCopyAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithCopy_whenCopyAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed",
        )

    def test_givenPtr_whenCallingReset_thenDestructs(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenPtr_whenCallingReset_thenDestructs"
        )

    def test_givenPtrWithCopy_whenCallingReset_thenDestructsAfterCopyDestructed(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithCopy_whenCallingReset_thenDestructsAfterCopyDestructed",
        )

    def test_givenPtrWithCopy_whenCallingResetOnCopy_thenDestructsAfterOriginalDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithCopy_whenCallingResetOnCopy_thenDestructsAfterOriginalDestructed",
        )

    def test_givenPtrWithMoved_whenCallingReset_thenDestructsAfterMovedDestructed(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithMoved_whenCallingReset_thenDestructsAfterMovedDestructed",
        )

    def test_givenPtrWithMoved_whenCallingResetOnMoved_thenDestructsImmediately(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithMoved_whenCallingResetOnMoved_thenDestructsImmediately",
        )

    def test_AllowsMoveConstructingToConst(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllowsMoveConstructingToConst")

    def test_AllowsCopyConstructingToConst(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllowsCopyConstructingToConst")

    def test_AllowsMoveAssigningToConst(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllowsMoveAssigningToConst")

    def test_AllowsCopyAssigningToConst(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllowsCopyAssigningToConst")

    def test_givenNewPtr_thenHasUseCount1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenNewPtr_thenHasUseCount1")

    def test_givenNewPtr_thenIsUnique(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenNewPtr_thenIsUnique")

    def test_givenEmptyPtr_thenHasUseCount0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenEmptyPtr_thenHasUseCount0")

    def test_givenEmptyPtr_thenIsNotUnique(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenEmptyPtr_thenIsNotUnique")

    def test_givenResetPtr_thenHasUseCount0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenResetPtr_thenHasUseCount0")

    def test_givenResetPtr_thenIsNotUnique(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenResetPtr_thenIsNotUnique")

    def test_givenMoveConstructedPtr_thenHasUseCount1(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveConstructedPtr_thenHasUseCount1"
        )

    def test_givenMoveConstructedPtr_thenIsUnique(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenMoveConstructedPtr_thenIsUnique")

    def test_givenMoveConstructedPtr_thenOldHasUseCount0(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveConstructedPtr_thenOldHasUseCount0"
        )

    def test_givenMoveConstructedPtr_thenOldIsNotUnique(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveConstructedPtr_thenOldIsNotUnique"
        )

    def test_givenMoveAssignedPtr_thenHasUseCount1(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveAssignedPtr_thenHasUseCount1"
        )

    def test_givenMoveAssignedPtr_thenIsUnique(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenMoveAssignedPtr_thenIsUnique")

    def test_givenMoveAssignedPtr_thenOldHasUseCount0(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveAssignedPtr_thenOldHasUseCount0"
        )

    def test_givenMoveAssignedPtr_thenOldIsNotUnique(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveAssignedPtr_thenOldIsNotUnique"
        )

    def test_givenCopyConstructedPtr_thenHasUseCount2(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenCopyConstructedPtr_thenHasUseCount2"
        )

    def test_givenCopyConstructedPtr_thenIsNotUnique(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenCopyConstructedPtr_thenIsNotUnique"
        )

    def test_givenCopyConstructedPtr_thenOldHasUseCount2(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenCopyConstructedPtr_thenOldHasUseCount2"
        )

    def test_givenCopyConstructedPtr_thenOldIsNotUnique(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenCopyConstructedPtr_thenOldIsNotUnique"
        )

    def test_givenCopyConstructedPtr_whenDestructingCopy_thenHasUseCount1(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenCopyConstructedPtr_whenDestructingCopy_thenHasUseCount1",
        )

    def test_givenCopyConstructedPtr_whenDestructingCopy_thenIsUnique(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenCopyConstructedPtr_whenDestructingCopy_thenIsUnique",
        )

    def test_givenCopyConstructedPtr_whenReassigningCopy_thenHasUseCount1(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenCopyConstructedPtr_whenReassigningCopy_thenHasUseCount1",
        )

    def test_givenCopyConstructedPtr_whenReassigningCopy_thenIsUnique(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenCopyConstructedPtr_whenReassigningCopy_thenIsUnique",
        )

    def test_givenCopyAssignedPtr_thenHasUseCount2(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenCopyAssignedPtr_thenHasUseCount2"
        )

    def test_givenCopyAssignedPtr_thenIsNotUnique(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenCopyAssignedPtr_thenIsNotUnique")

    def test_givenCopyAssignedPtr_whenDestructingCopy_thenHasUseCount1(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenCopyAssignedPtr_whenDestructingCopy_thenHasUseCount1",
        )

    def test_givenCopyAssignedPtr_whenDestructingCopy_thenIsUnique(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenCopyAssignedPtr_whenDestructingCopy_thenIsUnique",
        )

    def test_givenCopyAssignedPtr_whenReassigningCopy_thenHasUseCount1(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenCopyAssignedPtr_whenReassigningCopy_thenHasUseCount1",
        )

    def test_givenCopyAssignedPtr_whenReassigningCopy_thenIsUnique(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenCopyAssignedPtr_whenReassigningCopy_thenIsUnique",
        )

    def test_givenPtr_whenReleasedAndReclaimed_thenDoesntCrash(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenReleasedAndReclaimed_thenDoesntCrash",
        )

    def test_givenPtr_whenReleasedAndReclaimed_thenIsDestructedAtEnd(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenReleasedAndReclaimed_thenIsDestructedAtEnd",
        )

    def test_givenPtr_whenNonOwningReclaimed_thenDoesntCrash(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenNonOwningReclaimed_thenDoesntCrash",
        )

    def test_givenPtr_whenNonOwningReclaimed_thenIsDestructedAtEnd(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenNonOwningReclaimed_thenIsDestructedAtEnd",
        )


class TestWeakIntrusivePtrTest(TestCase):
    cpp_name = "WeakIntrusivePtrTest"

    def test_givenPtr_whenCreatingAndDestructing_thenDoesntCrash(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCreatingAndDestructing_thenDoesntCrash",
        )

    def test_givenPtr_whenLocking_thenReturnsCorrectObject(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenPtr_whenLocking_thenReturnsCorrectObject"
        )

    def test_expiredPtr_whenLocking_thenReturnsNullType(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "expiredPtr_whenLocking_thenReturnsNullType"
        )

    def test_weakNullPtr_locking(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "weakNullPtr_locking")

    def test_givenValidPtr_whenMoveAssigning_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigning_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenMoveAssigning_thenOldInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigning_thenOldInstanceInvalid",
        )

    def test_vector_insert_weak_intrusive(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "vector_insert_weak_intrusive")

    def test_givenInvalidPtr_whenMoveAssigning_thenNewInstanceIsValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenMoveAssigning_thenNewInstanceIsValid",
        )

    def test_givenValidPtr_whenMoveAssigningToSelf_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigningToSelf_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenMoveAssigningToSelf_thenStaysValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigningToSelf_thenStaysValid",
        )

    def test_givenInvalidPtr_whenMoveAssigning_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenMoveAssigning_thenPointsToSameObject",
        )

    def test_givenInvalidPtr_whenMoveAssigningToSelf_thenStaysInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenMoveAssigningToSelf_thenStaysInvalid",
        )

    def test_givenWeakOnlyPtr_whenMoveAssigning_thenNewInstanceIsValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenMoveAssigning_thenNewInstanceIsValid",
        )

    def test_givenWeakOnlyPtr_whenMoveAssigning_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenMoveAssigning_thenPointsToSameObject",
        )

    def test_givenWeakOnlyPtr_whenMoveAssigningToSelf_thenStaysInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenMoveAssigningToSelf_thenStaysInvalid",
        )

    def test_givenWeakOnlyPtr_whenMoveAssigningToSelf_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenMoveAssigningToSelf_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenMoveAssigningFromInvalidPtr_thenNewInstanceIsInvalid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigningFromInvalidPtr_thenNewInstanceIsInvalid",
        )

    def test_givenValidPtr_whenMoveAssigningFromWeakOnlyPtr_thenNewInstanceIsInvalid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigningFromWeakOnlyPtr_thenNewInstanceIsInvalid",
        )

    def test_givenValidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenMoveAssigningToBaseClass_thenOldInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenMoveAssigningToBaseClass_thenOldInstanceInvalid",
        )

    def test_givenInvalidPtr_whenMoveAssigningToBaseClass_thenNewInstanceIsValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenMoveAssigningToBaseClass_thenNewInstanceIsValid",
        )

    def test_givenInvalidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject",
        )

    def test_givenInvalidPtr_whenMoveAssigningInvalidPtrToBaseClass_thenNewInstanceIsValid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenMoveAssigningInvalidPtrToBaseClass_thenNewInstanceIsValid",
        )

    def test_givenWeakOnlyPtr_whenMoveAssigningToBaseClass_thenNewInstanceIsValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenMoveAssigningToBaseClass_thenNewInstanceIsValid",
        )

    def test_givenWeakOnlyPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject",
        )

    def test_givenWeakOnlyPtr_whenMoveAssigningInvalidPtrToBaseClass_thenNewInstanceIsValid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenMoveAssigningInvalidPtrToBaseClass_thenNewInstanceIsValid",
        )

    def test_givenNullPtr_whenMoveAssigningToDifferentNullptr_thenHasNewNullptr(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNullPtr_whenMoveAssigningToDifferentNullptr_thenHasNewNullptr",
        )

    def test_givenValidPtr_whenCopyAssigning_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenCopyAssigning_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenCopyAssigning_thenOldInstanceValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenCopyAssigning_thenOldInstanceValid",
        )

    def test_givenValidPtr_whenCopyAssigningToSelf_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenCopyAssigningToSelf_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenCopyAssigningToSelf_thenStaysValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenCopyAssigningToSelf_thenStaysValid",
        )

    def test_givenInvalidPtr_whenCopyAssigning_thenNewInstanceIsValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenCopyAssigning_thenNewInstanceIsValid",
        )

    def test_givenInvalidPtr_whenCopyAssigningToSelf_thenStaysInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenCopyAssigningToSelf_thenStaysInvalid",
        )

    def test_givenWeakOnlyPtr_whenCopyAssigning_thenNewInstanceIsValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenCopyAssigning_thenNewInstanceIsValid",
        )

    def test_givenWeakOnlyPtr_whenCopyAssigning_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenCopyAssigning_thenPointsToSameObject",
        )

    def test_givenWeakOnlyPtr_whenCopyAssigningToSelf_thenStaysInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenCopyAssigningToSelf_thenStaysInvalid",
        )

    def test_givenWeakOnlyPtr_whenCopyAssigningToSelf_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenCopyAssigningToSelf_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject",
        )

    def test_givenValidPtr_whenCopyAssigningToBaseClass_thenOldInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenValidPtr_whenCopyAssigningToBaseClass_thenOldInstanceInvalid",
        )

    def test_givenInvalidPtr_whenCopyAssigningToBaseClass_thenNewInstanceIsValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenCopyAssigningToBaseClass_thenNewInstanceIsValid",
        )

    def test_givenInvalidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInvalidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject",
        )

    def test_givenPtr_whenCopyAssigningInvalidPtrToBaseClass_thenNewInstanceIsInvalid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssigningInvalidPtrToBaseClass_thenNewInstanceIsInvalid",
        )

    def test_givenWeakOnlyPtr_whenCopyAssigningToBaseClass_thenNewInstanceIsValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenCopyAssigningToBaseClass_thenNewInstanceIsValid",
        )

    def test_givenWeakOnlyPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject",
        )

    def test_givenPtr_whenCopyAssigningWeakOnlyPtrToBaseClass_thenNewInstanceIsValid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssigningWeakOnlyPtrToBaseClass_thenNewInstanceIsValid",
        )

    def test_givenNullPtr_whenCopyAssigningToDifferentNullptr_thenHasNewNullptr(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNullPtr_whenCopyAssigningToDifferentNullptr_thenHasNewNullptr",
        )

    def test_givenPtr_whenMoveConstructing_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructing_thenPointsToSameObject",
        )

    def test_givenPtr_whenMoveConstructing_thenOldInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructing_thenOldInstanceInvalid",
        )

    def test_givenPtr_whenMoveConstructing_thenNewInstanceValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructing_thenNewInstanceValid",
        )

    def test_givenPtr_whenMoveConstructingFromInvalidPtr_thenNewInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructingFromInvalidPtr_thenNewInstanceInvalid",
        )

    def test_givenPtr_whenMoveConstructingFromWeakOnlyPtr_thenNewInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructingFromWeakOnlyPtr_thenNewInstanceInvalid",
        )

    def test_givenPtr_whenMoveConstructingToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructingToBaseClass_thenPointsToSameObject",
        )

    def test_givenPtr_whenMoveConstructingToBaseClass_thenOldInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructingToBaseClass_thenOldInstanceInvalid",
        )

    def test_givenPtr_whenMoveConstructingToBaseClass_thenNewInstanceValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructingToBaseClass_thenNewInstanceValid",
        )

    def test_givenPtr_whenMoveConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid",
        )

    def test_givenPtr_whenMoveConstructingToBaseClassFromWeakOnlyPtr_thenNewInstanceInvalid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructingToBaseClassFromWeakOnlyPtr_thenNewInstanceInvalid",
        )

    def test_givenNullPtr_whenMoveConstructingToDifferentNullptr_thenHasNewNullptr(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNullPtr_whenMoveConstructingToDifferentNullptr_thenHasNewNullptr",
        )

    def test_givenPtr_whenCopyConstructing_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructing_thenPointsToSameObject",
        )

    def test_givenPtr_whenCopyConstructing_thenOldInstanceValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructing_thenOldInstanceValid",
        )

    def test_givenPtr_whenCopyConstructing_thenNewInstanceValid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructing_thenNewInstanceValid",
        )

    def test_givenPtr_whenCopyConstructingFromInvalidPtr_thenNewInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructingFromInvalidPtr_thenNewInstanceInvalid",
        )

    def test_givenPtr_whenCopyConstructingFromWeakOnlyPtr_thenNewInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructingFromWeakOnlyPtr_thenNewInstanceInvalid",
        )

    def test_givenPtr_whenCopyConstructingToBaseClass_thenPointsToSameObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructingToBaseClass_thenPointsToSameObject",
        )

    def test_givenPtr_whenCopyConstructingToBaseClass_thenOldInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructingToBaseClass_thenOldInstanceInvalid",
        )

    def test_givenPtr_whenCopyConstructingToBaseClass_thenNewInstanceInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructingToBaseClass_thenNewInstanceInvalid",
        )

    def test_givenPtr_whenCopyConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid",
        )

    def test_givenPtr_whenCopyConstructingToBaseClassFromWeakOnlyPtr_thenNewInstanceInvalid(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructingToBaseClassFromWeakOnlyPtr_thenNewInstanceInvalid",
        )

    def test_givenNullPtr_whenCopyConstructingToDifferentNullptr_thenHasNewNullptr(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNullPtr_whenCopyConstructingToDifferentNullptr_thenHasNewNullptr",
        )

    def test_SwapFunction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapFunction")

    def test_SwapMethod(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapMethod")

    def test_SwapFunctionFromInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapFunctionFromInvalid")

    def test_SwapMethodFromInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapMethodFromInvalid")

    def test_SwapFunctionWithInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapFunctionWithInvalid")

    def test_SwapMethodWithInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapMethodWithInvalid")

    def test_SwapFunctionInvalidWithInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapFunctionInvalidWithInvalid")

    def test_SwapMethodInvalidWithInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapMethodInvalidWithInvalid")

    def test_SwapFunctionFromWeakOnlyPtr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapFunctionFromWeakOnlyPtr")

    def test_SwapMethodFromWeakOnlyPtr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapMethodFromWeakOnlyPtr")

    def test_SwapFunctionWithWeakOnlyPtr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapFunctionWithWeakOnlyPtr")

    def test_SwapMethodWithWeakOnlyPtr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapMethodWithWeakOnlyPtr")

    def test_SwapFunctionWeakOnlyPtrWithWeakOnlyPtr(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "SwapFunctionWeakOnlyPtrWithWeakOnlyPtr"
        )

    def test_SwapMethodWeakOnlyPtrWithWeakOnlyPtr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SwapMethodWeakOnlyPtrWithWeakOnlyPtr")

    def test_CanBePutInContainer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanBePutInContainer")

    def test_CanBePutInSet(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanBePutInSet")

    def test_CanBePutInUnorderedSet(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanBePutInUnorderedSet")

    def test_CanBePutInMap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanBePutInMap")

    def test_CanBePutInUnorderedMap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanBePutInUnorderedMap")

    def test_Equality_AfterCopyConstructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Equality_AfterCopyConstructor")

    def test_Equality_AfterCopyAssignment(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Equality_AfterCopyAssignment")

    def test_Equality_AfterCopyAssignment_WeakOnly(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "Equality_AfterCopyAssignment_WeakOnly"
        )

    def test_Equality_Invalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Equality_Invalid")

    def test_Inequality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Inequality")

    def test_Inequality_InvalidLeft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Inequality_InvalidLeft")

    def test_Inequality_InvalidRight(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Inequality_InvalidRight")

    def test_Inequality_WeakOnly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Inequality_WeakOnly")

    def test_HashIsDifferent(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashIsDifferent")

    def test_HashIsDifferent_ValidAndInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashIsDifferent_ValidAndInvalid")

    def test_HashIsDifferent_ValidAndWeakOnly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashIsDifferent_ValidAndWeakOnly")

    def test_HashIsDifferent_WeakOnlyAndWeakOnly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashIsDifferent_WeakOnlyAndWeakOnly")

    def test_HashIsSame_AfterCopyConstructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashIsSame_AfterCopyConstructor")

    def test_HashIsSame_AfterCopyConstructor_WeakOnly(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "HashIsSame_AfterCopyConstructor_WeakOnly"
        )

    def test_HashIsSame_AfterCopyAssignment(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashIsSame_AfterCopyAssignment")

    def test_HashIsSame_AfterCopyAssignment_WeakOnly(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "HashIsSame_AfterCopyAssignment_WeakOnly"
        )

    def test_HashIsSame_BothInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashIsSame_BothInvalid")

    def test_OneIsLess(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OneIsLess")

    def test_InvalidIsLess1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InvalidIsLess1")

    def test_InvalidIsLess2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InvalidIsLess2")

    def test_InvalidIsNotLessThanInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InvalidIsNotLessThanInvalid")

    def test_givenPtr_whenCallingResetOnWeakPtr_thenIsInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCallingResetOnWeakPtr_thenIsInvalid",
        )

    def test_givenPtr_whenCallingResetOnStrongPtr_thenIsInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCallingResetOnStrongPtr_thenIsInvalid",
        )

    def test_AllowsMoveConstructingToConst(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllowsMoveConstructingToConst")

    def test_AllowsCopyConstructingToConst(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllowsCopyConstructingToConst")

    def test_AllowsMoveAssigningToConst(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllowsMoveAssigningToConst")

    def test_AllowsCopyAssigningToConst(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllowsCopyAssigningToConst")

    def test_givenNewPtr_thenHasUseCount1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenNewPtr_thenHasUseCount1")

    def test_givenNewPtr_thenIsNotExpired(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenNewPtr_thenIsNotExpired")

    def test_givenInvalidPtr_thenHasUseCount0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenInvalidPtr_thenHasUseCount0")

    def test_givenInvalidPtr_thenIsExpired(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenInvalidPtr_thenIsExpired")

    def test_givenWeakOnlyPtr_thenHasUseCount0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenWeakOnlyPtr_thenHasUseCount0")

    def test_givenWeakOnlyPtr_thenIsExpired(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenWeakOnlyPtr_thenIsExpired")

    def test_givenPtr_whenCallingWeakReset_thenHasUseCount0(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenPtr_whenCallingWeakReset_thenHasUseCount0"
        )

    def test_givenPtr_whenCallingWeakReset_thenIsExpired(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenPtr_whenCallingWeakReset_thenIsExpired"
        )

    def test_givenPtr_whenCallingStrongReset_thenHasUseCount0(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCallingStrongReset_thenHasUseCount0",
        )

    def test_givenPtr_whenCallingStrongReset_thenIsExpired(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenPtr_whenCallingStrongReset_thenIsExpired"
        )

    def test_givenMoveConstructedPtr_thenHasUseCount1(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveConstructedPtr_thenHasUseCount1"
        )

    def test_givenMoveConstructedPtr_thenIsNotExpired(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveConstructedPtr_thenIsNotExpired"
        )

    def test_givenMoveConstructedPtr_thenOldHasUseCount0(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveConstructedPtr_thenOldHasUseCount0"
        )

    def test_givenMoveConstructedPtr_thenOldIsExpired(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveConstructedPtr_thenOldIsExpired"
        )

    def test_givenMoveAssignedPtr_thenHasUseCount1(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveAssignedPtr_thenHasUseCount1"
        )

    def test_givenMoveAssignedPtr_thenIsNotExpired(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveAssignedPtr_thenIsNotExpired"
        )

    def test_givenMoveAssignedPtr_thenOldHasUseCount0(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveAssignedPtr_thenOldHasUseCount0"
        )

    def test_givenMoveAssignedPtr_thenOldIsExpired(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenMoveAssignedPtr_thenOldIsExpired"
        )

    def test_givenCopyConstructedPtr_thenHasUseCount1(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenCopyConstructedPtr_thenHasUseCount1"
        )

    def test_givenCopyConstructedPtr_thenIsNotExpired(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenCopyConstructedPtr_thenIsNotExpired"
        )

    def test_givenCopyConstructedPtr_thenOldHasUseCount1(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenCopyConstructedPtr_thenOldHasUseCount1"
        )

    def test_givenCopyConstructedPtr_thenOldIsNotExpired(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenCopyConstructedPtr_thenOldIsNotExpired"
        )

    def test_givenPtr_whenLastStrongPointerResets_thenReleasesResources(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenLastStrongPointerResets_thenReleasesResources",
        )

    def test_givenPtr_whenDestructedButStillHasStrongPointers_thenDoesntReleaseResources(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenDestructedButStillHasStrongPointers_thenDoesntReleaseResources",
        )

    def test_givenPtr_whenDestructed_thenDestructsObject(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenPtr_whenDestructed_thenDestructsObject"
        )

    def test_givenPtr_whenMoveConstructed_thenDestructsObjectAfterSecondDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructed_thenDestructsObjectAfterSecondDestructed",
        )

    def test_givenPtr_whenMoveConstructedToBaseClass_thenDestructsObjectAfterSecondDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveConstructedToBaseClass_thenDestructsObjectAfterSecondDestructed",
        )

    def test_givenPtr_whenMoveAssigned_thenDestructsOldObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveAssigned_thenDestructsOldObject",
        )

    def test_givenPtr_whenMoveAssignedToBaseClass_thenDestructsOldObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveAssignedToBaseClass_thenDestructsOldObject",
        )

    def test_givenPtrWithCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed",
        )

    def test_givenPtrWithBaseClassCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithBaseClassCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed",
        )

    def test_givenPtrWithCopy_whenMoveAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithCopy_whenMoveAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed",
        )

    def test_givenPtr_whenMoveAssigned_thenDestructsObjectAfterSecondDestructed(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveAssigned_thenDestructsObjectAfterSecondDestructed",
        )

    def test_givenPtr_whenMoveAssignedToBaseClass_thenDestructsObjectAfterSecondDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenMoveAssignedToBaseClass_thenDestructsObjectAfterSecondDestructed",
        )

    def test_givenPtr_whenCopyConstructedAndDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructedAndDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyConstructedToBaseClassAndDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructedToBaseClassAndDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyConstructedAndOriginalDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructedAndOriginalDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyConstructedToBaseClassAndOriginalDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyConstructedToBaseClassAndOriginalDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyAssignedAndDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssignedAndDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyAssignedToBaseClassAndDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssignedToBaseClassAndDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyAssignedAndOriginalDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssignedAndOriginalDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyAssignedToBaseClassAndOriginalDestructed_thenDestructsObjectAfterLastDestruction(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssignedToBaseClassAndOriginalDestructed_thenDestructsObjectAfterLastDestruction",
        )

    def test_givenPtr_whenCopyAssigned_thenDestructsOldObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssigned_thenDestructsOldObject",
        )

    def test_givenPtr_whenCopyAssignedToBaseClass_thenDestructsOldObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenCopyAssignedToBaseClass_thenDestructsOldObject",
        )

    def test_givenPtrWithCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed",
        )

    def test_givenPtrWithBaseClassCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithBaseClassCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed",
        )

    def test_givenPtrWithCopy_whenCopyAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithCopy_whenCopyAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed",
        )

    def test_givenPtr_whenCallingReset_thenDestructs(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenPtr_whenCallingReset_thenDestructs"
        )

    def test_givenPtrWithCopy_whenCallingReset_thenDestructsAfterCopyDestructed(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithCopy_whenCallingReset_thenDestructsAfterCopyDestructed",
        )

    def test_givenPtrWithCopy_whenCallingResetOnCopy_thenDestructsAfterOriginalDestructed(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithCopy_whenCallingResetOnCopy_thenDestructsAfterOriginalDestructed",
        )

    def test_givenPtrWithMoved_whenCallingReset_thenDestructsAfterMovedDestructed(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithMoved_whenCallingReset_thenDestructsAfterMovedDestructed",
        )

    def test_givenPtrWithMoved_whenCallingResetOnMoved_thenDestructsImmediately(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtrWithMoved_whenCallingResetOnMoved_thenDestructsImmediately",
        )

    def test_givenPtr_whenReleasedAndReclaimed_thenDoesntCrash(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenReleasedAndReclaimed_thenDoesntCrash",
        )

    def test_givenWeakOnlyPtr_whenReleasedAndReclaimed_thenDoesntCrash(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenReleasedAndReclaimed_thenDoesntCrash",
        )

    def test_givenPtr_whenReleasedAndReclaimed_thenIsDestructedAtEnd(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenPtr_whenReleasedAndReclaimed_thenIsDestructedAtEnd",
        )

    def test_givenWeakOnlyPtr_whenReleasedAndReclaimed_thenIsDestructedAtEnd(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenWeakOnlyPtr_whenReleasedAndReclaimed_thenIsDestructedAtEnd",
        )

    def test_givenStackObject_whenReclaimed_thenCrashes(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenStackObject_whenReclaimed_thenCrashes"
        )


if __name__ == "__main__":
    run_tests()
