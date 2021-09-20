import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_typeid_test"


class TestTypeMetaTest(TestCase):
    cpp_name = "TypeMetaTest"

    def test_TypeMetaStatic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TypeMetaStatic")

    def test_Names(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Names")

    def test_TypeMeta(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TypeMeta")

    def test_CtorDtorAndCopy(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CtorDtorAndCopy")

    def test_Float16IsNotUint16(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Float16IsNotUint16")


if __name__ == "__main__":
    run_tests()
