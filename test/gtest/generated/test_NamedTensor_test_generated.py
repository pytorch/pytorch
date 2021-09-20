import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/NamedTensor_test"


class TestNamedTensorTest(TestCase):
    cpp_name = "NamedTensorTest"

    def test_isNamed(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "isNamed")

    def test_attachMetadata(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "attachMetadata")

    def test_internalSetNamesInplace(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "internalSetNamesInplace")

    def test_empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "empty")

    def test_dimnameToPosition(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "dimnameToPosition")

    def test_unifyFromRight(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "unifyFromRight")

    def test_alias(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "alias")

    def test_NoNamesGuard(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoNamesGuard")

    def test_TensorNamePrint(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorNamePrint")

    def test_TensorNamesCheckUnique(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorNamesCheckUnique")


if __name__ == "__main__":
    run_tests()
