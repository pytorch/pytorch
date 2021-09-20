import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/reduce_ops_test"


class TestReduceOpsTest(TestCase):
    cpp_name = "ReduceOpsTest"

    def test_MaxValuesAndMinValues(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxValuesAndMinValues")


if __name__ == "__main__":
    run_tests()
