import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/distributed_test"


class TestConverter(TestCase):
    cpp_name = "Converter"

    def test_DeclareExport(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeclareExport")

    def test_InjectDataEdgeIndicators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InjectDataEdgeIndicators")

    def test_OverloadedConvertToNNModule(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OverloadedConvertToNNModule")

    def test_OverloadedConvertToNNModuleFailure(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OverloadedConvertToNNModuleFailure")


class TestDistributed(TestCase):
    cpp_name = "Distributed"

    def test_InsertDeviceOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InsertDeviceOptions")

    def test_InsertDeviceOptionsFailureCase(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InsertDeviceOptionsFailureCase")


if __name__ == "__main__":
    run_tests()
