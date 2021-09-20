import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_LeftRight_test"


class TestLeftRightTest(TestCase):
    cpp_name = "LeftRightTest"

    def test_givenInt_whenWritingAndReading_thenChangesArePresent(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInt_whenWritingAndReading_thenChangesArePresent",
        )

    def test_givenVector_whenWritingAndReading_thenChangesArePresent(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenVector_whenWritingAndReading_thenChangesArePresent",
        )

    def test_givenVector_whenWritingReturnsValue_thenValueIsReturned(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenVector_whenWritingReturnsValue_thenValueIsReturned",
        )

    def test_readsCanBeConcurrent(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "readsCanBeConcurrent")

    def test_writesCanBeConcurrentWithReads_readThenWrite(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "writesCanBeConcurrentWithReads_readThenWrite"
        )

    def test_writesCanBeConcurrentWithReads_writeThenRead(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "writesCanBeConcurrentWithReads_writeThenRead"
        )

    def test_writesCannotBeConcurrentWithWrites(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "writesCannotBeConcurrentWithWrites")

    def test_whenReadThrowsException_thenThrowsThrough(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenReadThrowsException_thenThrowsThrough"
        )

    def test_whenWriteThrowsException_thenThrowsThrough(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "whenWriteThrowsException_thenThrowsThrough"
        )

    def test_givenInt_whenWriteThrowsExceptionOnFirstCall_thenResetsToOldState(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInt_whenWriteThrowsExceptionOnFirstCall_thenResetsToOldState",
        )

    def test_givenInt_whenWriteThrowsExceptionOnSecondCall_thenKeepsNewState(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenInt_whenWriteThrowsExceptionOnSecondCall_thenKeepsNewState",
        )

    def test_givenVector_whenWriteThrowsException_thenResetsToOldState(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenVector_whenWriteThrowsException_thenResetsToOldState",
        )


if __name__ == "__main__":
    run_tests()
