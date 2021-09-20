import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/test_cpp_rpc"


class TestWireSerialize(TestCase):
    cpp_name = "WireSerialize"

    def test_Base(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Base")

    def test_RecopySparseTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RecopySparseTensors")

    def test_CloneSparseTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CloneSparseTensors")

    def test_Errors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Errors")

    def test_DISABLED_Sparse(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_Sparse")


class TestTestE2ETensorPipe(TestCase):
    cpp_name = "TestE2ETensorPipe"

    def test_TestTrainingLoop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestTrainingLoop")


class TestTensorpipeSerialize(TestCase):
    cpp_name = "TensorpipeSerialize"

    def test_Base(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Base")

    def test_RecopySparseTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RecopySparseTensors")

    def test_NoDeleterTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoDeleterTensors")


if __name__ == "__main__":
    run_tests()
