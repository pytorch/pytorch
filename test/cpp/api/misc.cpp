#include <gtest/gtest.h>

#include <torch/nn/init.h>
#include <torch/nn/modules/linear.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <test/cpp/api/support.h>

#include <functional>

void torch_warn_once() {
  // TORCH_WARN_ONCE("TORCH_WARN_ONCE");
}

void torch_warn() {
  TORCH_WARN("TORCH_WARN_MULTIPLE_TIMES");
}

int count_substr_occurrences(const std::string& str, const std::string& substr) {
  int count = 0;
  size_t pos = str.find(substr);

  while (pos != std::string::npos) {
    count++;
    pos = str.find(substr, pos + substr.size());
  }

  return count;
}

TEST(UtilsTest, WarnOnce) {
  {
    std::stringstream buffer;
    torch::test::CerrRedirect cerr_redirect(buffer.rdbuf());

    // torch_warn_once();
    // torch_warn_once();

    // check that there is only one "torch_warn_once" in the string
    // ASSERT_EQ(count_substr_occurrences(buffer.str(), "TORCH_WARN_ONCE"), 1);
  }
  {
    std::stringstream buffer;
    torch::test::CerrRedirect cerr_redirect(buffer.rdbuf());

    torch_warn();
    torch_warn();
    torch_warn();

    ASSERT_EQ(count_substr_occurrences(buffer.str(), "TORCH_WARN_MULTIPLE_TIMES"), 3);
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
