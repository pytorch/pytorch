import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_Bitset_test"


class TestBitsetTest(TestCase):
    cpp_name = "BitsetTest"

    def test_givenEmptyBitset_whenGettingBit_thenIsZero(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenEmptyBitset_whenGettingBit_thenIsZero"
        )

    def test_givenEmptyBitset_whenUnsettingBit_thenIsZero(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenEmptyBitset_whenUnsettingBit_thenIsZero"
        )

    def test_givenEmptyBitset_whenSettingAndUnsettingBit_thenIsZero(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenEmptyBitset_whenSettingAndUnsettingBit_thenIsZero",
        )

    def test_givenEmptyBitset_whenSettingBit_thenIsSet(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenEmptyBitset_whenSettingBit_thenIsSet"
        )

    def test_givenEmptyBitset_whenSettingBit_thenOthersStayUnset(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenEmptyBitset_whenSettingBit_thenOthersStayUnset",
        )

    def test_givenNonemptyBitset_whenSettingBit_thenIsSet(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "givenNonemptyBitset_whenSettingBit_thenIsSet"
        )

    def test_givenNonemptyBitset_whenSettingBit_thenOthersStayAtOldValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNonemptyBitset_whenSettingBit_thenOthersStayAtOldValue",
        )

    def test_givenNonemptyBitset_whenUnsettingBit_thenIsUnset(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNonemptyBitset_whenUnsettingBit_thenIsUnset",
        )

    def test_givenNonemptyBitset_whenUnsettingBit_thenOthersStayAtOldValue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenNonemptyBitset_whenUnsettingBit_thenOthersStayAtOldValue",
        )

    def test_givenEmptyBitset_whenCallingForEachBit_thenDoesntCall(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenEmptyBitset_whenCallingForEachBit_thenDoesntCall",
        )

    def test_givenBitsetWithOneBitSet_whenCallingForEachBit_thenCallsForEachBit(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBitsetWithOneBitSet_whenCallingForEachBit_thenCallsForEachBit",
        )

    def test_givenBitsetWithMultipleBitsSet_whenCallingForEachBit_thenCallsForEachBit(
        self,
    ):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "givenBitsetWithMultipleBitsSet_whenCallingForEachBit_thenCallsForEachBit",
        )


if __name__ == "__main__":
    run_tests()
