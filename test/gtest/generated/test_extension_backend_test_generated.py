import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/extension_backend_test"


class TestBackendExtensionTest(TestCase):
    cpp_name = "BackendExtensionTest"

    def test_TestRegisterOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestRegisterOp")


if __name__ == "__main__":
    run_tests()
