import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/Dict_test"


class TestDictTest(TestCase):
    cpp_name = "DictTest"

    def test_givenEmptyDict_whenCallingEmpty_thenReturnsTrue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenEmptyDict_whenCallingEmpty_thenReturnsTrue",
        )

    def test_givenNonemptyDict_whenCallingEmpty_thenReturnsFalse(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNonemptyDict_whenCallingEmpty_thenReturnsFalse",
        )

    def test_givenEmptyDict_whenCallingSize_thenReturnsZero(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenEmptyDict_whenCallingSize_thenReturnsZero"
        )

    def test_givenNonemptyDict_whenCallingSize_thenReturnsNumberOfElements(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNonemptyDict_whenCallingSize_thenReturnsNumberOfElements",
        )

    def test_givenNonemptyDict_whenCallingClear_thenIsEmpty(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenNonemptyDict_whenCallingClear_thenIsEmpty"
        )

    def test_whenInsertingNewKey_thenReturnsTrueAndIteratorToNewElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenInsertingNewKey_thenReturnsTrueAndIteratorToNewElement",
        )

    def test_whenInsertingExistingKey_thenReturnsFalseAndIteratorToExistingElement(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenInsertingExistingKey_thenReturnsFalseAndIteratorToExistingElement",
        )

    def test_whenInsertingExistingKey_thenDoesNotModifyDict(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenInsertingExistingKey_thenDoesNotModifyDict"
        )

    def test_whenInsertOrAssigningNewKey_thenReturnsTrueAndIteratorToNewElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenInsertOrAssigningNewKey_thenReturnsTrueAndIteratorToNewElement",
        )

    def test_whenInsertOrAssigningExistingKey_thenReturnsFalseAndIteratorToChangedElement(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenInsertOrAssigningExistingKey_thenReturnsFalseAndIteratorToChangedElement",
        )

    def test_whenInsertOrAssigningExistingKey_thenDoesModifyDict(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenInsertOrAssigningExistingKey_thenDoesModifyDict",
        )

    def test_givenEmptyDict_whenIterating_thenBeginIsEnd(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenEmptyDict_whenIterating_thenBeginIsEnd"
        )

    def test_givenMutableDict_whenIterating_thenFindsElements(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenMutableDict_whenIterating_thenFindsElements",
        )

    def test_givenMutableDict_whenIteratingWithForeach_thenFindsElements(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenMutableDict_whenIteratingWithForeach_thenFindsElements",
        )

    def test_givenConstDict_whenIterating_thenFindsElements(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenConstDict_whenIterating_thenFindsElements"
        )

    def test_givenConstDict_whenIteratingWithForeach_thenFindsElements(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenConstDict_whenIteratingWithForeach_thenFindsElements",
        )

    def test_givenIterator_thenCanModifyValue(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "givenIterator_thenCanModifyValue")

    def test_givenOneElementDict_whenErasingByIterator_thenDictIsEmpty(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOneElementDict_whenErasingByIterator_thenDictIsEmpty",
        )

    def test_givenOneElementDict_whenErasingByKey_thenReturnsOneAndDictIsEmpty(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOneElementDict_whenErasingByKey_thenReturnsOneAndDictIsEmpty",
        )

    def test_givenOneElementDict_whenErasingByNonexistingKey_thenReturnsZeroAndDictIsUnchanged(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenOneElementDict_whenErasingByNonexistingKey_thenReturnsZeroAndDictIsUnchanged",
        )

    def test_whenCallingAtWithExistingKey_thenReturnsCorrectElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingAtWithExistingKey_thenReturnsCorrectElement",
        )

    def test_whenCallingAtWithNonExistingKey_thenReturnsCorrectElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingAtWithNonExistingKey_thenReturnsCorrectElement",
        )

    def test_givenMutableDict_whenCallingFindOnExistingKey_thenFindsCorrectElement(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenMutableDict_whenCallingFindOnExistingKey_thenFindsCorrectElement",
        )

    def test_givenMutableDict_whenCallingFindOnNonExistingKey_thenReturnsEnd(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenMutableDict_whenCallingFindOnNonExistingKey_thenReturnsEnd",
        )

    def test_givenConstDict_whenCallingFindOnExistingKey_thenFindsCorrectElement(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenConstDict_whenCallingFindOnExistingKey_thenFindsCorrectElement",
        )

    def test_givenConstDict_whenCallingFindOnNonExistingKey_thenReturnsEnd(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenConstDict_whenCallingFindOnNonExistingKey_thenReturnsEnd",
        )

    def test_whenCallingContainsWithExistingKey_thenReturnsTrue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingContainsWithExistingKey_thenReturnsTrue",
        )

    def test_whenCallingContainsWithNonExistingKey_thenReturnsFalse(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCallingContainsWithNonExistingKey_thenReturnsFalse",
        )

    def test_whenCallingReserve_thenDoesntCrash(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenCallingReserve_thenDoesntCrash")

    def test_whenCopyConstructingDict_thenAreEqual(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenCopyConstructingDict_thenAreEqual"
        )

    def test_whenCopyAssigningDict_thenAreEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenCopyAssigningDict_thenAreEqual")

    def test_whenCopyingDict_thenAreEqual(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenCopyingDict_thenAreEqual")

    def test_whenMoveConstructingDict_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenMoveConstructingDict_thenNewIsCorrect"
        )

    def test_whenMoveAssigningDict_thenNewIsCorrect(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenMoveAssigningDict_thenNewIsCorrect"
        )

    def test_whenMoveConstructingDict_thenOldIsEmpty(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenMoveConstructingDict_thenOldIsEmpty"
        )

    def test_whenMoveAssigningDict_thenOldIsEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenMoveAssigningDict_thenOldIsEmpty")

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

    def test_givenIterator_whenWritingToValue_thenChangesValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenWritingToValue_thenChangesValue",
        )

    def test_isReferenceType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "isReferenceType")

    def test_copyHasSeparateStorage(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "copyHasSeparateStorage")

    def test_dictTensorAsKey(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "dictTensorAsKey")

    def test_dictEquality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "dictEquality")


class TestListTest_IValueBasedList(TestCase):
    cpp_name = "ListTest_IValueBasedList"

    def test_givenIterator_whenWritingToValueFromIterator_thenChangesValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenIterator_whenWritingToValueFromIterator_thenChangesValue",
        )


if __name__ == "__main__":
    run_tests()
