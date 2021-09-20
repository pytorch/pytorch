import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/cast_test"


class TestCastTest(TestCase):
    cpp_name = "CastTest"

    def test_GetCastDataType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetCastDataType")


if __name__ == "__main__":
    run_tests()
