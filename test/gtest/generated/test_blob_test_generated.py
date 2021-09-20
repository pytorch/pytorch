import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/blob_test"


class TestTensorCPUDeathTest_0(TestCase):
    cpp_name = "TensorCPUDeathTest/0"

    def test_CannotAccessRawDataWhenEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CannotAccessRawDataWhenEmpty")

    def test_CannotAccessDataWhenEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CannotAccessDataWhenEmpty")


class TestTensorCPUDeathTest_1(TestCase):
    cpp_name = "TensorCPUDeathTest/1"

    def test_CannotAccessRawDataWhenEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CannotAccessRawDataWhenEmpty")

    def test_CannotAccessDataWhenEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CannotAccessDataWhenEmpty")


class TestTensorCPUDeathTest_2(TestCase):
    cpp_name = "TensorCPUDeathTest/2"

    def test_CannotAccessRawDataWhenEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CannotAccessRawDataWhenEmpty")

    def test_CannotAccessDataWhenEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CannotAccessDataWhenEmpty")


class TestTensorDeathTest(TestCase):
    cpp_name = "TensorDeathTest"

    def test_CannotCastDownLargeDims(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CannotCastDownLargeDims")


class TestBlobTest(TestCase):
    cpp_name = "BlobTest"

    def test_Blob(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Blob")

    def test_BlobUninitialized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BlobUninitialized")

    def test_BlobWrongType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BlobWrongType")

    def test_BlobReset(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BlobReset")

    def test_BlobMove(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BlobMove")

    def test_BlobNonConstructible(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BlobNonConstructible")

    def test_BlobShareExternalPointer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BlobShareExternalPointer")

    def test_BlobShareExternalObject(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BlobShareExternalObject")

    def test_StringSerialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StringSerialization")

    def test_CastingMessage(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CastingMessage")


class TestTensorNonTypedTest(TestCase):
    cpp_name = "TensorNonTypedTest"

    def test_TensorChangeType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorChangeType")

    def test_NonDefaultConstructible(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NonDefaultConstructible")


class TestTensorCPUTest_0(TestCase):
    cpp_name = "TensorCPUTest/0"

    def test_TensorInitializedEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInitializedEmpty")

    def test_TensorInitializedNonEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInitializedNonEmpty")

    def test_TensorInitializedZeroDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInitializedZeroDim")

    def test_TensorResizeZeroDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorResizeZeroDim")

    def test_TensorInitializedScalar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInitializedScalar")

    def test_TensorAlias(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorAlias")

    def test_TensorShareDataRawPointer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorShareDataRawPointer")

    def test_TensorShareDataRawPointerWithMeta(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorShareDataRawPointerWithMeta")

    def test_TensorAliasCanUseDifferentShapes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorAliasCanUseDifferentShapes")

    def test_NoLongerAliassAfterNumelChanges(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoLongerAliassAfterNumelChanges")

    def test_NoLongerAliasAfterFreeMemory(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoLongerAliasAfterFreeMemory")

    def test_KeepOnShrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "KeepOnShrink")

    def test_MaxKeepOnShrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxKeepOnShrink")


class TestTensorCPUTest_1(TestCase):
    cpp_name = "TensorCPUTest/1"

    def test_TensorInitializedEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInitializedEmpty")

    def test_TensorInitializedNonEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInitializedNonEmpty")

    def test_TensorInitializedZeroDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInitializedZeroDim")

    def test_TensorResizeZeroDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorResizeZeroDim")

    def test_TensorInitializedScalar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInitializedScalar")

    def test_TensorAlias(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorAlias")

    def test_TensorShareDataRawPointer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorShareDataRawPointer")

    def test_TensorShareDataRawPointerWithMeta(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorShareDataRawPointerWithMeta")

    def test_TensorAliasCanUseDifferentShapes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorAliasCanUseDifferentShapes")

    def test_NoLongerAliassAfterNumelChanges(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoLongerAliassAfterNumelChanges")

    def test_NoLongerAliasAfterFreeMemory(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoLongerAliasAfterFreeMemory")

    def test_KeepOnShrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "KeepOnShrink")

    def test_MaxKeepOnShrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxKeepOnShrink")


class TestTensorCPUTest_2(TestCase):
    cpp_name = "TensorCPUTest/2"

    def test_TensorInitializedEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInitializedEmpty")

    def test_TensorInitializedNonEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInitializedNonEmpty")

    def test_TensorInitializedZeroDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInitializedZeroDim")

    def test_TensorResizeZeroDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorResizeZeroDim")

    def test_TensorInitializedScalar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorInitializedScalar")

    def test_TensorAlias(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorAlias")

    def test_TensorShareDataRawPointer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorShareDataRawPointer")

    def test_TensorShareDataRawPointerWithMeta(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorShareDataRawPointerWithMeta")

    def test_TensorAliasCanUseDifferentShapes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorAliasCanUseDifferentShapes")

    def test_NoLongerAliassAfterNumelChanges(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoLongerAliassAfterNumelChanges")

    def test_NoLongerAliasAfterFreeMemory(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoLongerAliasAfterFreeMemory")

    def test_KeepOnShrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "KeepOnShrink")

    def test_MaxKeepOnShrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxKeepOnShrink")


class TestTensorTest(TestCase):
    cpp_name = "TensorTest"

    def test_TensorNonFundamentalType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorNonFundamentalType")

    def test_TensorNonFundamentalTypeClone(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorNonFundamentalTypeClone")

    def test_Tensor64BitDimension(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Tensor64BitDimension")

    def test_UndefinedTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UndefinedTensor")

    def test_CopyAndAssignment(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CopyAndAssignment")

    def test_TensorSerialization_bool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_bool")

    def test_TensorSerialization_double(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_double")

    def test_TensorSerialization_float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_float")

    def test_TensorSerialization_int(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_int")

    def test_TensorSerialization_int8_t(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_int8_t")

    def test_TensorSerialization_int16_t(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_int16_t")

    def test_TensorSerialization_uint8_t(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_uint8_t")

    def test_TensorSerialization_uint16_t(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_uint16_t")

    def test_TensorSerialization_int64_t(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_int64_t")

    def test_TensorSerialization_CustomType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_CustomType")

    def test_Half(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Half")

    def test_TensorFactory(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorFactory")


class TestEmptyTensorTest(TestCase):
    cpp_name = "EmptyTensorTest"

    def test_TensorSerialization_bool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_bool")

    def test_TensorSerialization_double(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_double")

    def test_TensorSerialization_float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_float")

    def test_TensorSerialization_int(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_int")

    def test_TensorSerialization_int8_t(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_int8_t")

    def test_TensorSerialization_int16_t(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_int16_t")

    def test_TensorSerialization_uint8_t(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_uint8_t")

    def test_TensorSerialization_uint16_t(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_uint16_t")

    def test_TensorSerialization_int64_t(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorSerialization_int64_t")


class TestQTensorTest(TestCase):
    cpp_name = "QTensorTest"

    def test_QTensorSerialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "QTensorSerialization")


class TestTypedTensorTest_0(TestCase):
    cpp_name = "TypedTensorTest/0"

    def test_BigTensorSerialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BigTensorSerialization")


class TestTypedTensorTest_1(TestCase):
    cpp_name = "TypedTensorTest/1"

    def test_BigTensorSerialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BigTensorSerialization")


class TestTypedTensorTest_2(TestCase):
    cpp_name = "TypedTensorTest/2"

    def test_BigTensorSerialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BigTensorSerialization")


class TestTypedTensorTest_3(TestCase):
    cpp_name = "TypedTensorTest/3"

    def test_BigTensorSerialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BigTensorSerialization")


class TestTypedTensorTest_4(TestCase):
    cpp_name = "TypedTensorTest/4"

    def test_BigTensorSerialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BigTensorSerialization")


class TestTypedTensorTest_5(TestCase):
    cpp_name = "TypedTensorTest/5"

    def test_BigTensorSerialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BigTensorSerialization")


class TestTypedTensorTest_6(TestCase):
    cpp_name = "TypedTensorTest/6"

    def test_BigTensorSerialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BigTensorSerialization")


class TestTypedTensorTest_7(TestCase):
    cpp_name = "TypedTensorTest/7"

    def test_BigTensorSerialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BigTensorSerialization")


class TestTypedTensorTest_8(TestCase):
    cpp_name = "TypedTensorTest/8"

    def test_BigTensorSerialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BigTensorSerialization")


class TestContentChunks(TestCase):
    cpp_name = "ContentChunks"

    def test_Serialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Serialization")


class TestCustomChunkSize(TestCase):
    cpp_name = "CustomChunkSize"

    def test_BigTensorSerialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BigTensorSerialization")


class TestQTensor(TestCase):
    cpp_name = "QTensor"

    def test_QTensorSizingTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "QTensorSizingTest")


class TestTensorConstruction(TestCase):
    cpp_name = "TensorConstruction"

    def test_UninitializedCopyTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UninitializedCopyTest")

    def test_CopyConstructorTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CopyConstructorTest")

    def test_MoveAssignmentOpTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAssignmentOpTest")


class TestTensorSerialization(TestCase):
    cpp_name = "TensorSerialization"

    def test_MistakenlySerializingDtypeUninitializedTensor(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "MistakenlySerializingDtypeUninitializedTensor"
        )

    def test_TestCorrectness(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestCorrectness")


if __name__ == "__main__":
    run_tests()
