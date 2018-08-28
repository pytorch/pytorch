#include <iostream>
#include <string>
#include <catch.hpp>

#include <torch/jit.h>
#include <torch/tensor.h>

TEST_CASE("torch script") {
  SECTION("multiple functions") {
    auto module = torch::jit::compile(R"JIT(
      def test_mul(a, b):
        return a * b
      def test_relu(a, b):
        return torch.relu(a + b)
      def test_while(a, i):
        while i < 10:
          a += a
          i += 1
        return a
    )JIT");
    auto a = torch::ones(1);
    auto b = torch::ones(1);

    REQUIRE(1 == torch::jit::run(
      module, "test_mul", a, b)[0].toTensor().toCLong());

    REQUIRE(2 == torch::jit::run(
      module, "test_relu", a, b)[0].toTensor().toCLong());

    REQUIRE(0x200 == torch::jit::run(
      module, "test_while", a, b)[0].toTensor().toCLong());
  }
}
