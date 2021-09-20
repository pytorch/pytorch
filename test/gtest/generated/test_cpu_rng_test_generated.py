import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/cpu_rng_test"


class TestRNGTest(TestCase):
    cpp_name = "RNGTest"

    def test_RandomFromTo(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RandomFromTo")

    def test_Random(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Random")

    def test_Random64bits(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Random64bits")

    def test_Normal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Normal")

    def test_Normal_float_Tensor_out(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Normal_float_Tensor_out")

    def test_Normal_Tensor_float_out(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Normal_Tensor_float_out")

    def test_Normal_Tensor_Tensor_out(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Normal_Tensor_Tensor_out")

    def test_Normal_float_Tensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Normal_float_Tensor")

    def test_Normal_Tensor_float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Normal_Tensor_float")

    def test_Normal_Tensor_Tensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Normal_Tensor_Tensor")

    def test_Uniform(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Uniform")

    def test_Cauchy(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cauchy")

    def test_LogNormal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LogNormal")

    def test_Geometric(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Geometric")

    def test_Exponential(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Exponential")

    def test_Bernoulli_Tensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Bernoulli_Tensor")

    def test_Bernoulli_scalar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Bernoulli_scalar")

    def test_Bernoulli(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Bernoulli")

    def test_Bernoulli_2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Bernoulli_2")

    def test_Bernoulli_p(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Bernoulli_p")

    def test_Bernoulli_p_2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Bernoulli_p_2")

    def test_Bernoulli_out(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Bernoulli_out")


if __name__ == "__main__":
    run_tests()
