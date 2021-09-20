import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_TypeList_test"


class TestTypeListTest(TestCase):
    cpp_name = "TypeListTest"

    def test_MapTypesToValues_sametype(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MapTypesToValues_sametype")

    def test_MapTypesToValues_differenttypes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MapTypesToValues_differenttypes")

    def test_MapTypesToValues_members(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MapTypesToValues_members")

    def test_MapTypesToValues_empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MapTypesToValues_empty")


if __name__ == "__main__":
    run_tests()
