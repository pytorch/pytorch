import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_Metaprogramming_test"


class TestMetaprogrammingTest(TestCase):
    cpp_name = "MetaprogrammingTest"

    def test_ExtractArgByFilteredIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExtractArgByFilteredIndex")

    def test_ExtractArgByFilteredIndex_singleInput(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ExtractArgByFilteredIndex_singleInput"
        )

    def test_ExtractArgByFilteredIndex_movableOnly(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ExtractArgByFilteredIndex_movableOnly"
        )

    def test_ExtractArgByFilteredIndex_onlyCopiesIfNecessary(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ExtractArgByFilteredIndex_onlyCopiesIfNecessary",
        )

    def test_ExtractArgByFilteredIndex_onlyMovesIfNecessary(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ExtractArgByFilteredIndex_onlyMovesIfNecessary"
        )

    def test_ExtractArgByFilteredIndex_keepsLValueReferencesIntact(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ExtractArgByFilteredIndex_keepsLValueReferencesIntact",
        )

    def test_FilterMap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FilterMap")

    def test_FilterMap_emptyInput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FilterMap_emptyInput")

    def test_FilterMap_emptyOutput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FilterMap_emptyOutput")

    def test_FilterMap_movableOnly_byRValue(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FilterMap_movableOnly_byRValue")

    def test_FilterMap_movableOnly_byValue(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FilterMap_movableOnly_byValue")

    def test_FilterMap_onlyCopiesIfNecessary(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FilterMap_onlyCopiesIfNecessary")

    def test_FilterMap_onlyMovesIfNecessary_1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FilterMap_onlyMovesIfNecessary_1")

    def test_FilterMap_onlyMovesIfNecessary_2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FilterMap_onlyMovesIfNecessary_2")

    def test_TupleElements_subsetSelection(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleElements_subsetSelection")

    def test_TupleElements_reorderSelection(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleElements_reorderSelection")

    def test_TupleTake_nonemptyPrefix(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleTake_nonemptyPrefix")

    def test_TupleTake_fullPrefix(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleTake_fullPrefix")

    def test_TupleTake_negative(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleTake_negative")

    def test_TupleSlice_middle(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleSlice_middle")

    def test_TupleSlice_full(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleSlice_full")

    def test_TupleMap_simple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleMap_simple")

    def test_TupleMap_mapperTakesDifferentButConvertibleType(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "TupleMap_mapperTakesDifferentButConvertibleType",
        )

    def test_TupleMap_mapperTakesConstRef(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleMap_mapperTakesConstRef")

    def test_TupleMap_mapsToDifferentTypes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleMap_mapsToDifferentTypes")

    def test_TupleMap_differentiatesLRValueReferences(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TupleMap_differentiatesLRValueReferences"
        )

    def test_TupleMap_canWorkWithMovableOnlyType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleMap_canWorkWithMovableOnlyType")

    def test_TupleMap_doesntUnecessarilyCopyValues(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TupleMap_doesntUnecessarilyCopyValues"
        )

    def test_TupleMap_doesntUnecessarilyMoveValues(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TupleMap_doesntUnecessarilyMoveValues"
        )

    def test_TupleMap_canBeUsedWithAutoLambdas(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleMap_canBeUsedWithAutoLambdas")

    def test_TupleConcat_zerotuples(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleConcat_zerotuples")

    def test_TupleConcat_oneemptytuple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleConcat_oneemptytuple")

    def test_TupleConcat_onenonemptytuple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleConcat_onenonemptytuple")

    def test_TupleConcat_twotuples(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleConcat_twotuples")

    def test_TupleConcat_threetuples(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleConcat_threetuples")

    def test_TupleConcat_emptytupleatbeginning(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleConcat_emptytupleatbeginning")

    def test_TupleConcat_emptytupleinmiddle(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleConcat_emptytupleinmiddle")

    def test_TupleConcat_emptytupleatend(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TupleConcat_emptytupleatend")

    def test_TupleConcat_workswithreferencesandpointers(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TupleConcat_workswithreferencesandpointers"
        )

    def test_TupleConcat_worksWithMovableOnlyTypes(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TupleConcat_worksWithMovableOnlyTypes"
        )

    def test_TupleConcat_doesntCopyMoreThanNecessary(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TupleConcat_doesntCopyMoreThanNecessary"
        )


if __name__ == "__main__":
    run_tests()
