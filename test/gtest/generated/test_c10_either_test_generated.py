import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_either_test"


class TestEitherTest(TestCase):
    cpp_name = "EitherTest"

    def test_SpaceUsage(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SpaceUsage")

    def test_givenLeft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenLeft")

    def test_givenRight(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenRight")

    def test_givenMakeLeft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenMakeLeft")

    def test_givenMakeLeftWithSameType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenMakeLeftWithSameType")

    def test_givenMakeRight(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenMakeRight")

    def test_givenMakeRightWithSameType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenMakeRightWithSameType")

    def test_givenMovableOnlyMakeLeft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenMovableOnlyMakeLeft")

    def test_givenMovableOnlyMakeRight(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenMovableOnlyMakeRight")

    def test_givenMultiParamMakeLeft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenMultiParamMakeLeft")

    def test_givenMultiParamMakeRight(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenMultiParamMakeRight")

    def test_givenLeftCopyConstructedFromValue_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftCopyConstructedFromValue_thenNewIsCorrect",
        )

    def test_givenLeftCopyConstructedFromValue_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftCopyConstructedFromValue_thenOldIsCorrect",
        )

    def test_givenRightCopyConstructedFromValue_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightCopyConstructedFromValue_thenNewIsCorrect",
        )

    def test_givenRightCopyConstructedFromValue_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightCopyConstructedFromValue_thenOldIsCorrect",
        )

    def test_givenLeftMoveConstructedFromValue_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftMoveConstructedFromValue_thenNewIsCorrect",
        )

    def test_givenLeftMoveConstructedFromValue_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftMoveConstructedFromValue_thenOldIsCorrect",
        )

    def test_givenRightMoveConstructedFromValue_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightMoveConstructedFromValue_thenNewIsCorrect",
        )

    def test_givenRightMoveConstructedFromValue_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightMoveConstructedFromValue_thenOldIsCorrect",
        )

    def test_givenLeftCopyAssignedFromValue_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftCopyAssignedFromValue_thenNewIsCorrect",
        )

    def test_givenLeftCopyAssignedFromValue_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftCopyAssignedFromValue_thenOldIsCorrect",
        )

    def test_givenRightCopyAssignedFromValue_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightCopyAssignedFromValue_thenNewIsCorrect",
        )

    def test_givenRightCopyAssignedFromValue_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightCopyAssignedFromValue_thenOldIsCorrect",
        )

    def test_givenLeftMoveAssignedFromValue_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftMoveAssignedFromValue_thenNewIsCorrect",
        )

    def test_givenLeftMoveAssignedFromValue_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftMoveAssignedFromValue_thenOldIsCorrect",
        )

    def test_givenRightMoveAssignedFromValue_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightMoveAssignedFromValue_thenNewIsCorrect",
        )

    def test_givenRightMoveAssignedFromValue_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightMoveAssignedFromValue_thenOldIsCorrect",
        )

    def test_givenLeftCopyConstructed_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenLeftCopyConstructed_thenNewIsCorrect"
        )

    def test_givenLeftCopyConstructed_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenLeftCopyConstructed_thenOldIsCorrect"
        )

    def test_givenLeftCopyConstructed_withSameType_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftCopyConstructed_withSameType_thenNewIsCorrect",
        )

    def test_givenLeftCopyConstructed_withSameType_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftCopyConstructed_withSameType_thenOldIsCorrect",
        )

    def test_givenRightCopyConstructed_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenRightCopyConstructed_thenNewIsCorrect"
        )

    def test_givenRightCopyConstructed_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenRightCopyConstructed_thenOldIsCorrect"
        )

    def test_givenRightCopyConstructed_withSameType_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightCopyConstructed_withSameType_thenNewIsCorrect",
        )

    def test_givenRightCopyConstructed_withSameType_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightCopyConstructed_withSameType_thenOldIsCorrect",
        )

    def test_givenLeftMoveConstructed_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenLeftMoveConstructed_thenNewIsCorrect"
        )

    def test_givenLeftMoveConstructed_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenLeftMoveConstructed_thenOldIsCorrect"
        )

    def test_givenLeftMoveConstructed_withSameType_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftMoveConstructed_withSameType_thenNewIsCorrect",
        )

    def test_givenLeftMoveConstructed_withSameType_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftMoveConstructed_withSameType_thenOldIsCorrect",
        )

    def test_givenRightMoveConstructed_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenRightMoveConstructed_thenNewIsCorrect"
        )

    def test_givenRightMoveConstructed_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenRightMoveConstructed_thenOldIsCorrect"
        )

    def test_givenRightMoveConstructed_withSameType_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightMoveConstructed_withSameType_thenNewIsCorrect",
        )

    def test_givenRightMoveConstructed_withSameType_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightMoveConstructed_withSameType_thenOldIsCorrect",
        )

    def test_givenLeftCopyAssigned_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenLeftCopyAssigned_thenNewIsCorrect"
        )

    def test_givenLeftCopyAssigned_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenLeftCopyAssigned_thenOldIsCorrect"
        )

    def test_givenLeftCopyAssigned_withSameType_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftCopyAssigned_withSameType_thenNewIsCorrect",
        )

    def test_givenLeftCopyAssigned_withSameType_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftCopyAssigned_withSameType_thenOldIsCorrect",
        )

    def test_givenRightCopyAssigned_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenRightCopyAssigned_thenNewIsCorrect"
        )

    def test_givenRightCopyAssigned_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenRightCopyAssigned_thenOldIsCorrect"
        )

    def test_givenRightCopyAssigned_withSameType_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightCopyAssigned_withSameType_thenNewIsCorrect",
        )

    def test_givenRightCopyAssigned_withSameType_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightCopyAssigned_withSameType_thenOldIsCorrect",
        )

    def test_givenLeftMoveAssigned_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenLeftMoveAssigned_thenNewIsCorrect"
        )

    def test_givenLeftMoveAssigned_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenLeftMoveAssigned_thenOldIsCorrect"
        )

    def test_givenLeftMoveAssigned_withSameType_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftMoveAssigned_withSameType_thenNewIsCorrect",
        )

    def test_givenLeftMoveAssigned_withSameType_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenLeftMoveAssigned_withSameType_thenOldIsCorrect",
        )

    def test_givenRightMoveAssigned_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenRightMoveAssigned_thenNewIsCorrect"
        )

    def test_givenRightMoveAssigned_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenRightMoveAssigned_thenOldIsCorrect"
        )

    def test_givenRightMoveAssigned_withSameType_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightMoveAssigned_withSameType_thenNewIsCorrect",
        )

    def test_givenRightMoveAssigned_withSameType_thenOldIsCorrect(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenRightMoveAssigned_withSameType_thenOldIsCorrect",
        )

    def test_givenLeft_whenModified_thenValueIsChanged(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenLeft_whenModified_thenValueIsChanged"
        )

    def test_givenRight_whenModified_thenValueIsChanged(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenRight_whenModified_thenValueIsChanged"
        )

    def test_canEmplaceConstructLeft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "canEmplaceConstructLeft")

    def test_canEmplaceConstructRight(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "canEmplaceConstructRight")

    def test_givenEqualLefts_thenAreEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenEqualLefts_thenAreEqual")

    def test_givenEqualLefts_thenAreNotUnequal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenEqualLefts_thenAreNotUnequal")

    def test_givenEqualRights_thenAreEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenEqualRights_thenAreEqual")

    def test_givenEqualRights_thenAreNotUnequal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenEqualRights_thenAreNotUnequal")

    def test_givenLeftAndRight_thenAreNotEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenLeftAndRight_thenAreNotEqual")

    def test_givenLeftAndRight_thenAreUnequal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenLeftAndRight_thenAreUnequal")

    def test_OutputLeft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OutputLeft")

    def test_OutputRight(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OutputRight")

    def test_givenLeftAndRightWithSameType_thenAreNotEqual(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenLeftAndRightWithSameType_thenAreNotEqual"
        )

    def test_givenLeftAndRightWithSameType_thenAreUnequal(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenLeftAndRightWithSameType_thenAreUnequal"
        )


class TestEitherTest_Destructor(TestCase):
    cpp_name = "EitherTest_Destructor"

    def test_LeftDestructorIsCalled(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LeftDestructorIsCalled")

    def test_RightDestructorIsCalled(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RightDestructorIsCalled")

    def test_LeftDestructorIsCalledAfterCopying(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LeftDestructorIsCalledAfterCopying")

    def test_RightDestructorIsCalledAfterCopying(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RightDestructorIsCalledAfterCopying")

    def test_LeftDestructorIsCalledAfterMoving(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LeftDestructorIsCalledAfterMoving")

    def test_RightDestructorIsCalledAfterMoving(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RightDestructorIsCalledAfterMoving")

    def test_LeftDestructorIsCalledAfterAssignment(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "LeftDestructorIsCalledAfterAssignment"
        )

    def test_RightDestructorIsCalledAfterAssignment(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RightDestructorIsCalledAfterAssignment"
        )

    def test_LeftDestructorIsCalledAfterMoveAssignment(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "LeftDestructorIsCalledAfterMoveAssignment"
        )

    def test_RightDestructorIsCalledAfterMoveAssignment(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RightDestructorIsCalledAfterMoveAssignment"
        )


if __name__ == "__main__":
    run_tests()
