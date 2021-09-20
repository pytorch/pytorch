import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_SizesAndStrides_test"


class TestSizesAndStridesTest(TestCase):
    cpp_name = "SizesAndStridesTest"

    def test_DefaultConstructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DefaultConstructor")

    def test_SetSizes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetSizes")

    def test_Resize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Resize")

    def test_SetAtIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetAtIndex")

    def test_SetAtIterator(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetAtIterator")

    def test_SetViaData(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetViaData")

    def test_MoveConstructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveConstructor")

    def test_CopyConstructor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CopyConstructor")

    def test_CopyAssignmentSmallToSmall(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CopyAssignmentSmallToSmall")

    def test_MoveAssignmentSmallToSmall(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAssignmentSmallToSmall")

    def test_CopyAssignmentSmallToBig(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CopyAssignmentSmallToBig")

    def test_MoveAssignmentSmallToBig(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAssignmentSmallToBig")

    def test_CopyAssignmentBigToBig(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CopyAssignmentBigToBig")

    def test_MoveAssignmentBigToBig(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAssignmentBigToBig")

    def test_CopyAssignmentBigToSmall(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CopyAssignmentBigToSmall")

    def test_MoveAssignmentBigToSmall(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAssignmentBigToSmall")

    def test_CopyAssignmentSelf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CopyAssignmentSelf")

    def test_MoveAssignmentSelf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAssignmentSelf")


if __name__ == "__main__":
    run_tests()
