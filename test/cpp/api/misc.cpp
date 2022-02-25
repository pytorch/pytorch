#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

#include <functional>

using namespace torch::test;

void torch_warn_once_A() {
  TORCH_WARN_ONCE("warn once");
}

void torch_warn_once_B() {
  TORCH_WARN_ONCE("warn something else once");
}

void torch_warn() {
  TORCH_WARN("warn multiple times");
}

TEST(UtilsTest, WarnOnce) {
  {
    WarningCapture warnings;

    torch_warn_once_A();
    torch_warn_once_A();
    torch_warn_once_B();
    torch_warn_once_B();

    ASSERT_EQ(count_substr_occurrences(warnings.str(), "warn once"), 1);
    ASSERT_EQ(
        count_substr_occurrences(warnings.str(), "warn something else once"),
        1);
  }
  {
    WarningCapture warnings;

    torch_warn();
    torch_warn();
    torch_warn();

    ASSERT_EQ(
        count_substr_occurrences(warnings.str(), "warn multiple times"), 3);
  }
}

TEST(NoGradTest, SetsGradModeCorrectly) {
  torch::manual_seed(0);
  torch::NoGradGuard guard;
  torch::nn::Linear model(5, 2);
  auto x = torch::randn({10, 5}, torch::requires_grad());
  auto y = model->forward(x);
  torch::Tensor s = y.sum();

  // Mimicking python API behavior:
  ASSERT_THROWS_WITH(s.backward(),
    "element 0 of tensors does not require grad and does not have a grad_fn")
}

struct AutogradTest : torch::test::SeedingFixture {
  AutogradTest() {
    x = torch::randn({3, 3}, torch::requires_grad());
    y = torch::randn({3, 3});
    z = x * y;
  }
  torch::Tensor x, y, z;
};

TEST_F(AutogradTest, CanTakeDerivatives) {
  z.backward(torch::ones_like(z));
  ASSERT_TRUE(x.grad().allclose(y));
}

TEST_F(AutogradTest, CanTakeDerivativesOfZeroDimTensors) {
  z.sum().backward();
  ASSERT_TRUE(x.grad().allclose(y));
}

TEST_F(AutogradTest, CanPassCustomGradientInputs) {
  z.sum().backward(torch::ones({}) * 2);
  ASSERT_TRUE(x.grad().allclose(y * 2));
}

TEST(UtilsTest, AmbiguousOperatorDefaults) {
  auto tmp = at::empty({}, at::kCPU);
  at::_test_ambiguous_defaults(tmp);
  at::_test_ambiguous_defaults(tmp, 1);
  at::_test_ambiguous_defaults(tmp, 1, 1);
  at::_test_ambiguous_defaults(tmp, 2, "2");
}
