import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_registry_test"


class TestRegistryTest(TestCase):
    cpp_name = "RegistryTest"

    def test_CanRunCreator(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanRunCreator")

    def test_ReturnNullOnNonExistingCreator(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReturnNullOnNonExistingCreator")

    def test_RegistryPriorities(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegistryPriorities")


if __name__ == "__main__":
    run_tests()
