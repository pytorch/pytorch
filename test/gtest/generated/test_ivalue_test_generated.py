import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/ivalue_test"


class TestIValueTest(TestCase):
    cpp_name = "IValueTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_BasicStorage(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BasicStorage")

    def test_ComplexDict(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ComplexDict")

    def test_Swap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Swap")

    def test_CopyConstruct(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CopyConstruct")

    def test_MoveConstruct(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveConstruct")

    def test_CopyAssign(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CopyAssign")

    def test_MoveAssign(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAssign")

    def test_Tuple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Tuple")

    def test_unsafeRemoveAttr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "unsafeRemoveAttr")

    def test_TuplePrint(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TuplePrint")

    def test_ComplexIValuePrint(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ComplexIValuePrint")

    def test_Complex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Complex")

    def test_BasicFuture(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BasicFuture")

    def test_FutureCallbacks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FutureCallbacks")

    def test_FutureExceptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FutureExceptions")

    def test_FutureSetError(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FutureSetError")

    def test_ValueEquality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ValueEquality")

    def test_TensorEquality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorEquality")

    def test_ListEquality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ListEquality")

    def test_DictEquality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DictEquality")

    def test_DictEqualityDifferentOrder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DictEqualityDifferentOrder")

    def test_ListNestedEquality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ListNestedEquality")

    def test_StreamEquality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StreamEquality")

    def test_EnumEquality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EnumEquality")

    def test_isPtrType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "isPtrType")

    def test_isAliasOf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "isAliasOf")

    def test_internalToPointer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "internalToPointer")

    def test_IdentityComparisonAndHashing(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IdentityComparisonAndHashing")

    def test_getSubValues(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "getSubValues")

    def test_ScalarBool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScalarBool")

    def test_ToWeakAndBack(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ToWeakAndBack")


if __name__ == "__main__":
    run_tests()
