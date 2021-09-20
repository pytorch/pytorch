import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/time_observer_test"


class TestTimeObserverTest(TestCase):
    cpp_name = "TimeObserverTest"

    def test_Test3Seconds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Test3Seconds")


if __name__ == "__main__":
    run_tests()
