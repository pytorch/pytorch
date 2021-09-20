import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/cpu_generator_test"


class TestCPUGeneratorImpl(TestCase):
    cpp_name = "CPUGeneratorImpl"

    def test_TestGeneratorDynamicCast(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestGeneratorDynamicCast")

    def test_TestDefaultGenerator(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDefaultGenerator")

    def test_TestCloning(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestCloning")

    def test_TestMultithreadingGetEngineOperator(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestMultithreadingGetEngineOperator")

    def test_TestGetSetCurrentSeed(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestGetSetCurrentSeed")

    def test_TestMultithreadingGetSetCurrentSeed(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestMultithreadingGetSetCurrentSeed")

    def test_TestRNGForking(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestRNGForking")

    def test_TestPhiloxEngineReproducibility(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestPhiloxEngineReproducibility")

    def test_TestPhiloxEngineOffset1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestPhiloxEngineOffset1")

    def test_TestPhiloxEngineOffset2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestPhiloxEngineOffset2")

    def test_TestPhiloxEngineOffset3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestPhiloxEngineOffset3")

    def test_TestPhiloxEngineIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestPhiloxEngineIndex")

    def test_TestMT19937EngineReproducibility(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestMT19937EngineReproducibility")


if __name__ == "__main__":
    run_tests()
