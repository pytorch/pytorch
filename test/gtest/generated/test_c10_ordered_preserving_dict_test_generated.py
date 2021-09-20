import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_ordered_preserving_dict_test"


class TestOrderedPreservingDictTest(TestCase):
    cpp_name = "OrderedPreservingDictTest"

    def test_InsertAndDeleteBasic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InsertAndDeleteBasic")

    def test_InsertExistingDoesntAffectOrder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InsertExistingDoesntAffectOrder")

    def test_testRefType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "testRefType")

    def test_DictCollisions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DictCollisions")

    def test_test_range_insert(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "test_range_insert")

    def test_test_range_erase_all(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "test_range_erase_all")

    def test_test_range_erase(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "test_range_erase")

    def test_test_move_constructor_empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "test_move_constructor_empty")

    def test_test_move_operator_empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "test_move_operator_empty")

    def test_test_reassign_moved_object_move_constructor(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "test_reassign_moved_object_move_constructor"
        )

    def test_test_reassign_moved_object_move_operator(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "test_reassign_moved_object_move_operator"
        )

    def test_test_copy_constructor_and_operator(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "test_copy_constructor_and_operator")

    def test_test_copy_constructor_empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "test_copy_constructor_empty")

    def test_test_copy_operator_empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "test_copy_operator_empty")

    def test_test_at(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "test_at")

    def test_test_equal_range(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "test_equal_range")

    def test_test_access_operator(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "test_access_operator")

    def test_test_swap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "test_swap")

    def test_test_swap_empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "test_swap_empty")


if __name__ == "__main__":
    run_tests()
