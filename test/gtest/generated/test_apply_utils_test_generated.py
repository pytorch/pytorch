import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/apply_utils_test"


class TestApplyUtilsTest(TestCase):
    cpp_name = "ApplyUtilsTest"

    def test_Contiguous2D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Contiguous2D")

    def test_Small2D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Small2D")

    def test__2D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "_2D")

    def test__3D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "_3D")

    def test_Medium3D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Medium3D")

    def test__10D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "_10D")


if __name__ == "__main__":
    run_tests()
