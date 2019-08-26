#include <gtest/gtest.h>

#include <torch/nn/init.h>
#include <torch/nn/modules/linear.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <test/cpp/api/support.h>

#include <functional>

using namespace torch::test;

TEST(UtilsTest, WarnOnce) {
  {
    std::stringstream buffer;
    CerrRedirect cerr_redirect(buffer.rdbuf());

    TORCH_WARN_ONCE("warn once");
    TORCH_WARN_ONCE("warn something else once");

    ASSERT_EQ(count_substr_occurrences(buffer.str(), "warn once"), 1);
    ASSERT_EQ(count_substr_occurrences(buffer.str(), "warn something else once"), 1);
  }
  {
    std::stringstream buffer;
    CerrRedirect cerr_redirect(buffer.rdbuf());

    TORCH_WARN("warn multiple times");
    TORCH_WARN("warn multiple times");
    TORCH_WARN("warn multiple times");

    ASSERT_EQ(count_substr_occurrences(buffer.str(), "warn multiple times"), 3);
  }
}

TEST(NoGradTest, SetsGradModeCorrectly) {
  torch::manual_seed(0);
  torch::NoGradGuard guard;
  torch::nn::Linear model(5, 2);
  auto x = torch::randn({10, 5}, torch::requires_grad());
  auto y = model->forward(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_FALSE(model->weight.grad().defined());
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
  z.backward();
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
