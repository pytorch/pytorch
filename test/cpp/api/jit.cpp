#include <gtest/gtest.h>

#include <torch/jit.h>
#include <torch/types.h>

#include <string>

TEST(TorchScriptTest, CanCompileMultipleFunctions) {
  auto module = torch::jit::compile(R"JIT(
      def test_mul(a, b):
        return a * b
      def test_relu(a, b):
        return torch.relu(a + b)
      def test_while(a, i):
        while bool(i < 10):
          a += a
          i += 1
        return a
    )JIT");
  auto a = torch::ones(1);
  auto b = torch::ones(1);

  ASSERT_EQ(1, module->run_method("test_mul", a, b).toTensor().item<int64_t>());

  ASSERT_EQ(2, module->run_method("test_relu", a, b).toTensor().item<int64_t>());

  ASSERT_TRUE(
      0x200 == module->run_method("test_while", a, b).toTensor().item<int64_t>());
}
