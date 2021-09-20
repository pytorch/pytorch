import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/module_test"


class TestModuleTest(TestCase):
    cpp_name = "ModuleTest"

    def test_StaticModule(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StaticModule")

    def test_DynamicModule(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DynamicModule")


if __name__ == "__main__":
    run_tests()
