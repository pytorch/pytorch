#include <gtest/gtest.h>

#include <torch/nn/init.h>
#include <torch/nn/modules/linear.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <test/cpp/api/support.h>

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

TEST_F(AutogradTest, CannotModifyVersionInVarTypeMode) {
  y.add_(1);
  ASSERT_THROWS_WITH(z.sum().backward(), "modified by an inplace operation");
}

TEST_F(AutogradTest, CannotModifyVersionInNonVarTypeMode) {
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    y.add_(1);  // This uses non-Variable type dispatch
  }
  ASSERT_THROWS_WITH(z.sum().backward(), "modified by an inplace operation");
}
