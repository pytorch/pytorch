import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_string_view_test"


class TestStringViewTest(TestCase):
    cpp_name = "StringViewTest"

    def test_testStringConstructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testStringConstructor")

    def test_testConversionToString(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testConversionToString")

    def test_testCopyAssignment(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testCopyAssignment")

    def test_whenCallingAccessOperatorOutOfRange_thenThrows(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenCallingAccessOperatorOutOfRange_thenThrows"
        )

    def test_whenRemovingValidPrefix_thenWorks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenRemovingValidPrefix_thenWorks")

    def test_whenRemovingTooLargePrefix_thenThrows(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenRemovingTooLargePrefix_thenThrows"
        )

    def test_whenRemovingValidSuffix_thenWorks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenRemovingValidSuffix_thenWorks")

    def test_whenRemovingTooLargeSuffix_thenThrows(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenRemovingTooLargeSuffix_thenThrows"
        )

    def test_testSwapFunction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testSwapFunction")

    def test_testSwapMethod(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testSwapMethod")

    def test_whenCopyingFullStringView_thenDestinationHasCorrectData(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCopyingFullStringView_thenDestinationHasCorrectData",
        )

    def test_whenCopyingSubstr_thenDestinationHasCorrectData(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "whenCopyingSubstr_thenDestinationHasCorrectData",
        )

    def test_whenCopyingTooMuch_thenJustCopiesLess(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenCopyingTooMuch_thenJustCopiesLess"
        )

    def test_whenCopyingJustAtRange_thenDoesntCrash(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenCopyingJustAtRange_thenDoesntCrash"
        )

    def test_whenCopyingOutOfRange_thenThrows(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenCopyingOutOfRange_thenThrows")

    def test_whenCallingSubstrWithPosOutOfRange_thenThrows(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenCallingSubstrWithPosOutOfRange_thenThrows"
        )

    def test_testOutputOperator(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testOutputOperator")

    def test_testHash(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testHash")


if __name__ == "__main__":
    run_tests()
