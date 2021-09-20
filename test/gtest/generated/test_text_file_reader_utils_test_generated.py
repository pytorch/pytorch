import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/text_file_reader_utils_test"


class TestTextFileReaderUtilsTest(TestCase):
    cpp_name = "TextFileReaderUtilsTest"

    def test_TokenizeTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TokenizeTest")


if __name__ == "__main__":
    run_tests()
