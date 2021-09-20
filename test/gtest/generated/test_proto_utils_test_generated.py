import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/proto_utils_test"


class TestProtoUtilsTest(TestCase):
    cpp_name = "ProtoUtilsTest"

    def test_IsSameDevice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsSameDevice")

    def test_SimpleReadWrite(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimpleReadWrite")

    def test_CleanupExternalInputsAndOutputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CleanupExternalInputsAndOutputs")


if __name__ == "__main__":
    run_tests()
