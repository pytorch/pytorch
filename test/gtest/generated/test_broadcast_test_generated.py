import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/broadcast_test"


class TestBroadcastTest(TestCase):
    cpp_name = "BroadcastTest"

    def test_Broadcast(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Broadcast")


if __name__ == "__main__":
    run_tests()
