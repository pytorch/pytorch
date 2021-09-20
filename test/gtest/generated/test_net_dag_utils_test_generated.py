import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/net_dag_utils_test"


class TestDagUtilTest(TestCase):
    cpp_name = "DagUtilTest"

    def test_Empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Empty")

    def test_AllSync(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllSync")

    def test_AllAsync(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllAsync")

    def test_Mixed0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Mixed0")

    def test_Mixed1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Mixed1")

    def test_Mixed2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Mixed2")


if __name__ == "__main__":
    run_tests()
