import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/parallel_net_test"


class TestDAGNetTest(TestCase):
    cpp_name = "DAGNetTest"

    def test_TestDAGNetTiming(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDAGNetTiming")

    def test_TestDAGNetTimingReadAfterRead(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDAGNetTimingReadAfterRead")

    def test_TestDAGNetTimingWriteAfterWrite(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDAGNetTimingWriteAfterWrite")

    def test_TestDAGNetTimingWriteAfterRead(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDAGNetTimingWriteAfterRead")

    def test_TestDAGNetTimingControlDependency(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDAGNetTimingControlDependency")


class TestSimpleNetTest(TestCase):
    cpp_name = "SimpleNetTest"

    def test_TestSimpleNetTiming(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSimpleNetTiming")

    def test_TestSimpleNetTimingReadAfterRead(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSimpleNetTimingReadAfterRead")

    def test_TestSimpleNetTimingWriteAfterWrite(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSimpleNetTimingWriteAfterWrite")

    def test_TestSimpleNetTimingWriteAfterRead(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSimpleNetTimingWriteAfterRead")

    def test_TestSimpleNetTimingControlDependency(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSimpleNetTimingControlDependency")


if __name__ == "__main__":
    run_tests()
