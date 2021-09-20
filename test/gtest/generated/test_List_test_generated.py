import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/List_test"


class TestListTest_IValueBasedList(TestCase):
    cpp_name = "ListTest_IValueBasedList"

    def test_givenEmptyList_whenCallingEmpty_thenReturnsTrue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenEmptyList_whenCallingEmpty_thenReturnsTrue",
        )

    def test_givenNonemptyList_whenCallingEmpty_thenReturnsFalse(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNonemptyList_whenCallingEmpty_thenReturnsFalse",
        )

    def test_givenEmptyList_whenCallingSize_thenReturnsZero(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenEmptyList_whenCallingSize_thenReturnsZero"
        )

    def test_givenNonemptyList_whenCallingSize_thenReturnsNumberOfElements(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNonemptyList_whenCallingSize_thenReturnsNumberOfElements",
        )

    def test_givenNonemptyList_whenCallingClear_thenIsEmpty(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenNonemptyList_whenCallingClear_thenIsEmpty"
        )

    def test_whenCallingGetWithExistingPosition_thenReturnsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingGetWithExistingPosition_thenReturnsElement",
        )

    def test_whenCallingGetWithNonExistingPosition_thenThrowsException(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingGetWithNonExistingPosition_thenThrowsException",
        )

    def test_whenCallingExtractWithExistingPosition_thenReturnsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingExtractWithExistingPosition_thenReturnsElement",
        )

    def test_whenCallingExtractWithExistingPosition_thenListElementBecomesInvalid(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingExtractWithExistingPosition_thenListElementBecomesInvalid",
        )

    def test_whenCallingExtractWithNonExistingPosition_thenThrowsException(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingExtractWithNonExistingPosition_thenThrowsException",
        )

    def test_whenCallingCopyingSetWithExistingPosition_thenChangesElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingCopyingSetWithExistingPosition_thenChangesElement",
        )

    def test_whenCallingMovingSetWithExistingPosition_thenChangesElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingMovingSetWithExistingPosition_thenChangesElement",
        )

    def test_whenCallingCopyingSetWithNonExistingPosition_thenThrowsException(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingCopyingSetWithNonExistingPosition_thenThrowsException",
        )

    def test_whenCallingMovingSetWithNonExistingPosition_thenThrowsException(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingMovingSetWithNonExistingPosition_thenThrowsException",
        )

    def test_whenCallingAccessOperatorWithExistingPosition_thenReturnsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingAccessOperatorWithExistingPosition_thenReturnsElement",
        )

    def test_whenAssigningToAccessOperatorWithExistingPosition_thenSetsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenAssigningToAccessOperatorWithExistingPosition_thenSetsElement",
        )

    def test_whenAssigningToAccessOperatorFromAccessOperator_thenSetsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenAssigningToAccessOperatorFromAccessOperator_thenSetsElement",
        )

    def test_whenSwappingFromAccessOperator_thenSwapsElements(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenSwappingFromAccessOperator_thenSwapsElements",
        )

    def test_whenCallingAccessOperatorWithNonExistingPosition_thenThrowsException(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingAccessOperatorWithNonExistingPosition_thenThrowsException",
        )

    def test_whenCallingInsertOnIteratorWithLValue_thenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingInsertOnIteratorWithLValue_thenInsertsElement",
        )

    def test_whenCallingInsertOnIteratorWithRValue_thenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingInsertOnIteratorWithRValue_thenInsertsElement",
        )

    def test_whenCallingInsertWithLValue_thenReturnsIteratorToNewElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingInsertWithLValue_thenReturnsIteratorToNewElement",
        )

    def test_whenCallingInsertWithRValue_thenReturnsIteratorToNewElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingInsertWithRValue_thenReturnsIteratorToNewElement",
        )

    def test_whenCallingEmplaceWithLValue_thenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingEmplaceWithLValue_thenInsertsElement",
        )

    def test_whenCallingEmplaceWithRValue_thenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingEmplaceWithRValue_thenInsertsElement",
        )

    def test_whenCallingEmplaceWithConstructorArg_thenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingEmplaceWithConstructorArg_thenInsertsElement",
        )

    def test_whenCallingPushBackWithLValue_ThenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingPushBackWithLValue_ThenInsertsElement",
        )

    def test_whenCallingPushBackWithRValue_ThenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingPushBackWithRValue_ThenInsertsElement",
        )

    def test_whenCallingEmplaceBackWithLValue_ThenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingEmplaceBackWithLValue_ThenInsertsElement",
        )

    def test_whenCallingEmplaceBackWithRValue_ThenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingEmplaceBackWithRValue_ThenInsertsElement",
        )

    def test_whenCallingEmplaceBackWithConstructorArg_ThenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingEmplaceBackWithConstructorArg_ThenInsertsElement",
        )

    def test_givenEmptyList_whenIterating_thenBeginIsEnd(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenEmptyList_whenIterating_thenBeginIsEnd"
        )

    def test_whenIterating_thenFindsElements(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenIterating_thenFindsElements")

    def test_whenIteratingWithForeach_thenFindsElements(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenIteratingWithForeach_thenFindsElements"
        )

    def test_givenOneElementList_whenErasing_thenListIsEmpty(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOneElementList_whenErasing_thenListIsEmpty",
        )

    def test_givenList_whenErasing_thenReturnsIterator(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenList_whenErasing_thenReturnsIterator"
        )

    def test_givenList_whenErasingFullRange_thenIsEmpty(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenList_whenErasingFullRange_thenIsEmpty"
        )

    def test_whenCallingReserve_thenDoesntCrash(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenCallingReserve_thenDoesntCrash")

    def test_whenCopyConstructingList_thenAreEqual(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenCopyConstructingList_thenAreEqual"
        )

    def test_whenCopyAssigningList_thenAreEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenCopyAssigningList_thenAreEqual")

    def test_whenCopyingList_thenAreEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenCopyingList_thenAreEqual")

    def test_whenMoveConstructingList_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenMoveConstructingList_thenNewIsCorrect"
        )

    def test_whenMoveAssigningList_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenMoveAssigningList_thenNewIsCorrect"
        )

    def test_whenMoveConstructingList_thenOldIsEmpty(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenMoveConstructingList_thenOldIsEmpty"
        )

    def test_whenMoveAssigningList_thenOldIsEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenMoveAssigningList_thenOldIsEmpty")

    def test_givenIterator_whenPostfixIncrementing_thenMovesToNextAndReturnsOldPosition(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenPostfixIncrementing_thenMovesToNextAndReturnsOldPosition",
        )

    def test_givenIterator_whenPrefixIncrementing_thenMovesToNextAndReturnsNewPosition(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenPrefixIncrementing_thenMovesToNextAndReturnsNewPosition",
        )

    def test_givenIterator_whenPostfixDecrementing_thenMovesToNextAndReturnsOldPosition(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenPostfixDecrementing_thenMovesToNextAndReturnsOldPosition",
        )

    def test_givenIterator_whenPrefixDecrementing_thenMovesToNextAndReturnsNewPosition(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenPrefixDecrementing_thenMovesToNextAndReturnsNewPosition",
        )

    def test_givenIterator_whenIncreasing_thenMovesToNextAndReturnsNewPosition(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenIncreasing_thenMovesToNextAndReturnsNewPosition",
        )

    def test_givenIterator_whenDecreasing_thenMovesToNextAndReturnsNewPosition(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenDecreasing_thenMovesToNextAndReturnsNewPosition",
        )

    def test_givenIterator_whenAdding_thenReturnsNewIterator(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenAdding_thenReturnsNewIterator",
        )

    def test_givenIterator_whenSubtracting_thenReturnsNewIterator(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenSubtracting_thenReturnsNewIterator",
        )

    def test_givenIterator_whenCalculatingDifference_thenReturnsCorrectNumber(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenCalculatingDifference_thenReturnsCorrectNumber",
        )

    def test_givenEqualIterators_thenAreEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenEqualIterators_thenAreEqual")

    def test_givenDifferentIterators_thenAreNotEqual(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenDifferentIterators_thenAreNotEqual"
        )

    def test_givenIterator_whenDereferencing_thenPointsToCorrectElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenDereferencing_thenPointsToCorrectElement",
        )

    def test_givenIterator_whenAssigningNewValue_thenChangesValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenAssigningNewValue_thenChangesValue",
        )

    def test_givenIterator_whenAssigningNewValueFromIterator_thenChangesValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenAssigningNewValueFromIterator_thenChangesValue",
        )

    def test_givenIterator_whenSwappingValuesFromIterator_thenChangesValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenSwappingValuesFromIterator_thenChangesValue",
        )

    def test_givenOneElementList_whenCallingPopBack_thenIsEmpty(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOneElementList_whenCallingPopBack_thenIsEmpty",
        )

    def test_givenEmptyList_whenCallingResize_thenResizesAndSetsEmptyValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenEmptyList_whenCallingResize_thenResizesAndSetsEmptyValue",
        )

    def test_givenEmptyList_whenCallingResizeWithValue_thenResizesAndSetsValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenEmptyList_whenCallingResizeWithValue_thenResizesAndSetsValue",
        )

    def test_isReferenceType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "isReferenceType")

    def test_copyHasSeparateStorage(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "copyHasSeparateStorage")

    def test_givenEqualLists_thenIsEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenEqualLists_thenIsEqual")

    def test_givenDifferentLists_thenIsNotEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenDifferentLists_thenIsNotEqual")


class TestListTest_NonIValueBasedList(TestCase):
    cpp_name = "ListTest_NonIValueBasedList"

    def test_givenEmptyList_whenCallingEmpty_thenReturnsTrue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenEmptyList_whenCallingEmpty_thenReturnsTrue",
        )

    def test_givenNonemptyList_whenCallingEmpty_thenReturnsFalse(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNonemptyList_whenCallingEmpty_thenReturnsFalse",
        )

    def test_givenEmptyList_whenCallingSize_thenReturnsZero(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenEmptyList_whenCallingSize_thenReturnsZero"
        )

    def test_givenNonemptyList_whenCallingSize_thenReturnsNumberOfElements(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNonemptyList_whenCallingSize_thenReturnsNumberOfElements",
        )

    def test_givenNonemptyList_whenCallingClear_thenIsEmpty(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenNonemptyList_whenCallingClear_thenIsEmpty"
        )

    def test_whenCallingGetWithExistingPosition_thenReturnsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingGetWithExistingPosition_thenReturnsElement",
        )

    def test_whenCallingGetWithNonExistingPosition_thenThrowsException(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingGetWithNonExistingPosition_thenThrowsException",
        )

    def test_whenCallingExtractWithExistingPosition_thenReturnsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingExtractWithExistingPosition_thenReturnsElement",
        )

    def test_whenCallingExtractWithNonExistingPosition_thenThrowsException(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingExtractWithNonExistingPosition_thenThrowsException",
        )

    def test_whenCallingCopyingSetWithExistingPosition_thenChangesElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingCopyingSetWithExistingPosition_thenChangesElement",
        )

    def test_whenCallingMovingSetWithExistingPosition_thenChangesElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingMovingSetWithExistingPosition_thenChangesElement",
        )

    def test_whenCallingCopyingSetWithNonExistingPosition_thenThrowsException(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingCopyingSetWithNonExistingPosition_thenThrowsException",
        )

    def test_whenCallingMovingSetWithNonExistingPosition_thenThrowsException(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingMovingSetWithNonExistingPosition_thenThrowsException",
        )

    def test_whenCallingAccessOperatorWithExistingPosition_thenReturnsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingAccessOperatorWithExistingPosition_thenReturnsElement",
        )

    def test_whenAssigningToAccessOperatorWithExistingPosition_thenSetsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenAssigningToAccessOperatorWithExistingPosition_thenSetsElement",
        )

    def test_whenAssigningToAccessOperatorFromAccessOperator_thenSetsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenAssigningToAccessOperatorFromAccessOperator_thenSetsElement",
        )

    def test_whenSwappingFromAccessOperator_thenSwapsElements(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenSwappingFromAccessOperator_thenSwapsElements",
        )

    def test_whenCallingAccessOperatorWithNonExistingPosition_thenThrowsException(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingAccessOperatorWithNonExistingPosition_thenThrowsException",
        )

    def test_whenCallingInsertOnIteratorWithLValue_thenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingInsertOnIteratorWithLValue_thenInsertsElement",
        )

    def test_whenCallingInsertOnIteratorWithRValue_thenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingInsertOnIteratorWithRValue_thenInsertsElement",
        )

    def test_whenCallingInsertWithLValue_thenReturnsIteratorToNewElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingInsertWithLValue_thenReturnsIteratorToNewElement",
        )

    def test_whenCallingInsertWithRValue_thenReturnsIteratorToNewElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingInsertWithRValue_thenReturnsIteratorToNewElement",
        )

    def test_whenCallingEmplaceWithLValue_thenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingEmplaceWithLValue_thenInsertsElement",
        )

    def test_whenCallingEmplaceWithRValue_thenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingEmplaceWithRValue_thenInsertsElement",
        )

    def test_whenCallingEmplaceWithConstructorArg_thenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingEmplaceWithConstructorArg_thenInsertsElement",
        )

    def test_whenCallingPushBackWithLValue_ThenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingPushBackWithLValue_ThenInsertsElement",
        )

    def test_whenCallingPushBackWithRValue_ThenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingPushBackWithRValue_ThenInsertsElement",
        )

    def test_whenCallingEmplaceBackWithLValue_ThenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingEmplaceBackWithLValue_ThenInsertsElement",
        )

    def test_whenCallingEmplaceBackWithRValue_ThenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingEmplaceBackWithRValue_ThenInsertsElement",
        )

    def test_whenCallingEmplaceBackWithConstructorArg_ThenInsertsElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingEmplaceBackWithConstructorArg_ThenInsertsElement",
        )

    def test_givenEmptyList_whenIterating_thenBeginIsEnd(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenEmptyList_whenIterating_thenBeginIsEnd"
        )

    def test_whenIterating_thenFindsElements(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenIterating_thenFindsElements")

    def test_whenIteratingWithForeach_thenFindsElements(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenIteratingWithForeach_thenFindsElements"
        )

    def test_givenOneElementList_whenErasing_thenListIsEmpty(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOneElementList_whenErasing_thenListIsEmpty",
        )

    def test_givenList_whenErasing_thenReturnsIterator(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenList_whenErasing_thenReturnsIterator"
        )

    def test_givenList_whenErasingFullRange_thenIsEmpty(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenList_whenErasingFullRange_thenIsEmpty"
        )

    def test_whenCallingReserve_thenDoesntCrash(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenCallingReserve_thenDoesntCrash")

    def test_whenCopyConstructingList_thenAreEqual(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenCopyConstructingList_thenAreEqual"
        )

    def test_whenCopyAssigningList_thenAreEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenCopyAssigningList_thenAreEqual")

    def test_whenCopyingList_thenAreEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenCopyingList_thenAreEqual")

    def test_whenMoveConstructingList_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenMoveConstructingList_thenNewIsCorrect"
        )

    def test_whenMoveAssigningList_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenMoveAssigningList_thenNewIsCorrect"
        )

    def test_whenMoveConstructingList_thenOldIsEmpty(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenMoveConstructingList_thenOldIsEmpty"
        )

    def test_whenMoveAssigningList_thenOldIsEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenMoveAssigningList_thenOldIsEmpty")

    def test_givenIterator_whenPostfixIncrementing_thenMovesToNextAndReturnsOldPosition(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenPostfixIncrementing_thenMovesToNextAndReturnsOldPosition",
        )

    def test_givenIterator_whenPrefixIncrementing_thenMovesToNextAndReturnsNewPosition(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenPrefixIncrementing_thenMovesToNextAndReturnsNewPosition",
        )

    def test_givenIterator_whenPostfixDecrementing_thenMovesToNextAndReturnsOldPosition(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenPostfixDecrementing_thenMovesToNextAndReturnsOldPosition",
        )

    def test_givenIterator_whenPrefixDecrementing_thenMovesToNextAndReturnsNewPosition(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenPrefixDecrementing_thenMovesToNextAndReturnsNewPosition",
        )

    def test_givenIterator_whenIncreasing_thenMovesToNextAndReturnsNewPosition(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenIncreasing_thenMovesToNextAndReturnsNewPosition",
        )

    def test_givenIterator_whenDecreasing_thenMovesToNextAndReturnsNewPosition(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenDecreasing_thenMovesToNextAndReturnsNewPosition",
        )

    def test_givenIterator_whenAdding_thenReturnsNewIterator(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenAdding_thenReturnsNewIterator",
        )

    def test_givenIterator_whenSubtracting_thenReturnsNewIterator(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenSubtracting_thenReturnsNewIterator",
        )

    def test_givenIterator_whenCalculatingDifference_thenReturnsCorrectNumber(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenCalculatingDifference_thenReturnsCorrectNumber",
        )

    def test_givenEqualIterators_thenAreEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenEqualIterators_thenAreEqual")

    def test_givenDifferentIterators_thenAreNotEqual(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenDifferentIterators_thenAreNotEqual"
        )

    def test_givenIterator_whenDereferencing_thenPointsToCorrectElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenDereferencing_thenPointsToCorrectElement",
        )

    def test_givenIterator_whenAssigningNewValue_thenChangesValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenAssigningNewValue_thenChangesValue",
        )

    def test_givenIterator_whenAssigningNewValueFromIterator_thenChangesValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenAssigningNewValueFromIterator_thenChangesValue",
        )

    def test_givenIterator_whenSwappingValuesFromIterator_thenChangesValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenSwappingValuesFromIterator_thenChangesValue",
        )

    def test_givenOneElementList_whenCallingPopBack_thenIsEmpty(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOneElementList_whenCallingPopBack_thenIsEmpty",
        )

    def test_givenEmptyList_whenCallingResize_thenResizesAndSetsEmptyValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenEmptyList_whenCallingResize_thenResizesAndSetsEmptyValue",
        )

    def test_givenEmptyList_whenCallingResizeWithValue_thenResizesAndSetsValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenEmptyList_whenCallingResizeWithValue_thenResizesAndSetsValue",
        )

    def test_isReferenceType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "isReferenceType")

    def test_copyHasSeparateStorage(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "copyHasSeparateStorage")

    def test_givenEqualLists_thenIsEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenEqualLists_thenIsEqual")

    def test_givenDifferentLists_thenIsNotEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenDifferentLists_thenIsNotEqual")

    def test_isChecksIdentity(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "isChecksIdentity")

    def test_sameValueDifferentStorage_thenIsReturnsFalse(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "sameValueDifferentStorage_thenIsReturnsFalse"
        )


class TestListTest(TestCase):
    cpp_name = "ListTest"

    def test_canAccessStringByReference(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "canAccessStringByReference")

    def test_canAccessOptionalStringByReference(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "canAccessOptionalStringByReference")

    def test_canAccessTensorByReference(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "canAccessTensorByReference")


if __name__ == "__main__":
    run_tests()
