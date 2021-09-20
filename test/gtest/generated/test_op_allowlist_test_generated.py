import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/op_allowlist_test"

if __name__ == "__main__":
    run_tests()
