import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/inline_container_test"


class TestPyTorchStreamWriterAndReader(TestCase):
    cpp_name = "PyTorchStreamWriterAndReader"

    def test_SaveAndLoad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SaveAndLoad")


class TestPytorchStreamWriterAndReader(TestCase):
    cpp_name = "PytorchStreamWriterAndReader"

    def test_GetNonexistentRecordThrows(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetNonexistentRecordThrows")


if __name__ == "__main__":
    run_tests()
